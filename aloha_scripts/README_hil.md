# ALOHA 人在回路（HIL）部署与采集

机器人端统一用本仓库；openpi、N0-VTLA 只在 GPU 服务器上提供推理服务（websocket + msgpack，两者协议相同）。
接管方式是离合式（clutch）：暂停时主臂对齐到从臂，释放后等主臂稳定，记下 `offset = 主臂 - 从臂`，
接管期间从臂执行 `主臂 - offset`，所以主臂松手下坠的那一点不会传到从臂。数据按训练仓库原本读取的格式保存，
每帧额外带 HIL 标签。

启动步骤与按键说明见 `README_run_robot.md`；只验证模型（不接管、不录制）用 `run_pi_policy.py` / `run_vtla_policy.py`。

| 文件                      | 作用                                                                            |
| ------------------------- | ------------------------------------------------------------------------------- |
| `hil_rollout.py`        | 主脚本：推理执行、状态机、录制                                                  |
| `policy_client.py`      | websocket 客户端；openpi / N0 观测适配器（`run_vtla_policy.py` 也用它）       |
| `chunk_executor.py`     | 动作 chunk 执行：每 `--action_horizon` 步阻塞推理一次（同 openpi aloha_real） |
| `hil_devices.py`        | 离合主臂（P 增益渐降释放、稳定检测、offset）、从臂指令与安全限制、按键/踏板     |
| `hil_recorder.py`       | hdf5 / VTLA 两种格式的写入                                                      |
| `hil_split_episodes.py` | 把 HIL episode 切成只含接管段（或自主段）的 episode                             |
| `hil_compare_lengths.py` | 量原数据和接管数据的长度，算接管数据占原数据的百分比、要复制几份才到 10% / 20%  |
| `tests/`                | 不接硬件的测试                                                                  |

## 环境

机器人端在 conda 环境 `aloha`（Python 3.8）里运行，msgpack / websockets 等都已装好：`conda activate aloha`。

## 1. 服务器：起推理服务

openpi（服务器上，归一化统计按配置里的绝对路径读取 `/mnt/ssd2/guningquan/Dataset/packaging_simple_2_lerobot/norm_stats.json`，不用另外拷）：

```bash
cd /mnt/ssd2/guningquan/Programs_server/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_packaging_simple_2_full \
  --policy.dir=/mnt/ssd2/guningquan/Programs_server/openpi_checkpoints/pi05_packaging_simple_2_full/packaging_simple_2_pt/20000
```

N0-VTLA（见 `N0-VTLA/docs/DEPLOY.md`）：

```bash
cd /mnt/ssd2/guningquan/Programs_server/N0-VTLA
python scripts/serve_policy.py --policy.config=vtla_joint_posttrain --policy.dir=checkpoints/vtla_joint_posttrain/<exp>/<step>
```

## 2. 机器人：运行 HIL

第一条之前会发一次预热推理（openpi 首次推理要编译模型，可能几分钟，`--warmup_timeout` 默认 900 s）。

先起 ROS（`roslaunch aloha 4arms_teleop.launch`；N0 或要力反馈时用触觉 launch：
`roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:=force`，启动时它会给触觉清零，不要碰传感器）。

```bash
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2

# openpi pi0.5，保存 hdf5（默认 <dataset_dir>_hil，例如 .../dataset/packaging_simple_2_hil）
python3 aloha_scripts/hil_rollout.py --task_name packaging_simple_2 --policy openpi \
  --host 10.240.77.20 --port 8000 --prompt "packaging the items" --num_episodes 20

# N0-VTLA（joint 或 eef），保存 VTLA 原始目录
python3 aloha_scripts/hil_rollout.py --task_name aloha2_tactile_test --policy n0_joint \
  --host 10.240.77.20 --prompt "Perform the task."
```

### 按键

| 键        | 作用                                         |
| --------- | -------------------------------------------- |
| →        | 开始；之后暂停 / 恢复策略                    |
| ←        | 暂停时接管；接管中再按一次交还（回到暂停）   |
| e / Enter | 结束本条 → → 保存，← 丢弃，ESC 丢弃并退出 |
| ESC / q   | 结束本条，问完保存/丢弃后退出                |
| b         | 重新标定主臂夹爪力反馈零点                   |

踏板如果模拟的是方向键，直接可用；evdev 踏板用 `--pedal_device /dev/input/by-id/...`
（`--pedal_pause KEY_A --pedal_correct KEY_B --pedal_end KEY_C`）。

### 流程与参数

```
自主（主臂锁住）──→──▶ 暂停：从臂保持，主臂在 --align_s（2.0s）内移到从臂位置
暂停 ──←──▶ 主臂 P 增益在 --release_ramp_s（0.5s）内降到 0 → 断扭矩 → 恢复增益 → 力反馈开启
        → 等主臂稳定（--settle_s 0.5s 内每个关节变化 < --settle_tolerance 0.01rad，最多 --settle_timeout_s 2.0s）
        → offset = 主臂 - 从臂 → 接管：从臂 = 主臂 - offset（夹爪绝对值）；--correction_blend_s 默认 0（不缓入，第一条指令本来就等于保持的指令）
接管 ──←──▶ 主臂锁住、offset 清零 → 暂停          暂停 ──→──▶ 自主（--resume_blend_s，默认 0）
```

- 按 ← 之后到接管开始之前，进度条那一行（仍是黄色 `paused`）会显示准备进度，只显示两只主臂里更慢的那一只：
  - `releasing [██████░░░░] 7/12`：主臂正在变软（P 增益渐降），还不用动手。
  - `settling [████░░░░░░] steady 0.20/0.50s | move 0.014/0.010 rad (R elbow) | wait 0.6/2.0s`：手放稳。
    `steady` 是所有关节连续保持在 `--settle_tolerance` 以内的时间，涨到 `--settle_s` 就记 offset 并开始接管；
    某个关节变化超过 `--settle_tolerance` 时进度条清零，并标出是哪只手臂的哪个关节（`move` 为 `--settle_s` 窗口内最大变化，
    所以你停手后它要再过约 `--settle_s` 才回到阈值以下，这期间进度条在涨，属正常）；
    `wait` 是已等待时间，到 `--settle_timeout_s` 不再等，直接记 offset（日志里会写 `settle timeout`）。
  - 默认 `--settle_s 0.5` / `--settle_timeout_s 2.0`：0.2s 的窗口太短，进度条几乎看不出变化；超时必须不小于 `--settle_s`，
    否则机器人开场动作做完后才会报 `ValueError`。变红之前你动的部分会被吸收进 offset，从臂不会跟着动，所以先稳住再操作。
  - 颜色：这一段一直是黄色系，变红才表示已开始接管；绿色、红色不用于这个进度条（绿＝策略在跑，红＝你在接管）。
  - 只是显示，不影响控制逻辑，也不影响保存的数据。
- 每次接管的前 `--correction_skip_s`（0.2s）照常录制，但不标记为 intervention。
- 默认值与 lerobot 版命令一致（blend 0、skip 0.2、ramp 0.3）。打开缓入时从从臂上一条指令开始，不从实测位置开始（避免从臂先下沉）。
- 主臂变软时的寄存器写入在每条主臂自己的后台线程里执行：两条臂并行，控制循环不等待。
- `--absolute_takeover`：不用离合，从臂直接跟主臂绝对位置。
- 释放和稳定都在 50Hz 控制循环里按 tick 推进，不阻塞：这段时间从臂照常保持、照常录制。
- 安全：从臂目标先夹到关节限位，每 tick 关节变化不超过 `--max_joint_step`（0.1rad），
  相机数据超过 `--max_stream_age`（0.5s）、触觉帧本身超过 `--max_tactile_age`（0.2s，按发布器的 `local_read_end`
  算的数据年龄，不是收到消息的时间），或推理出错 / 超时（`--infer_timeout` 10s）时自动暂停。
- 用到触觉时（力反馈、`--format vtla`，或 hdf5 任务配了 `tactile_sensors`），每条 episode 开始前通过 `/tactile/calibrate`
  给触觉清零（从臂在起始姿态、夹爪张开，这时不要碰传感器），然后才测力反馈零点；`--no_tactile_calibration` 关闭。
  清零后导出的 runtime config 随 episode 保存。

## 3. 数据格式

- `--format hdf5`（openpi 默认）：`episode_N.hdf5`，布局与 `record_episodes_aloha2.py` 完全相同，另加
  `/hil/phase`（0 自主，1 暂停，2 接管）、`/hil/intervention`（0/1）、`/hil/leader_action`（主臂绝对读数，
  手臂关节上 `leader_action - action` 就是离合 offset）、`/hil/time`，以及根属性 `hil_*`（策略、prompt、参数、服务端 metadata）。
  和 `record_episodes_aloha2.py` 一样，还有每个 tick 的时间戳 `/observations/timestamps`；任务配了 `tactile_sensors`
  时有 `/observations/tactile`（和同一 tick 的相机图像一起取，触觉图像存 JPEG，`--tactile_jpeg_quality` 默认 90）和
  `/metadata/tactile`（runtime config）。
- `--format vtla`（N0 必须）：`episode_N/`，与 `record_episodes_aloha2_vtla.py` 相同，另加 `hil/data.csv`
  （与其它 CSV 同一个 30Hz 网格）和 `metadata.json` 里的 `hil`。触觉 runtime config 在 `tactile_runtime_config/`，
  `timestamps.csv` 末尾几列 `*_data_age_s` 是每行触觉的数据年龄。

两种格式都可以用 `python3 aloha_scripts/check_episode_timing.py <目录或文件>` 检查时间对齐和触觉是否新鲜。

完整 episode 可以直接交给各转换脚本（它们忽略 `/hil` 和 `hil/`）。只要接管段时先切分：

```bash
python3 aloha_scripts/hil_split_episodes.py <dataset>_hil <dataset>_hil_corrections --keep intervention
# --keep autonomous：只要策略执行段；--keep active：策略段 + 接管段，在暂停处切开；--min_frames 25
```

切分时 hdf5 里按时间排列的数据（包括时间戳和触觉）一起切，`/metadata/tactile` 和 VTLA 的
`tactile_runtime_config/` 每段原样保留。切分结果用原来的转换脚本即可（已验证）：openpi `convert_aloha_data_to_lerobot_no_bgr2rgb.py`（v2.1）、
lerobot `examples/aloha_ros/convert_aloha_hdf5_to_lerobot_v30.py`（v3.0）、N0 `convert_canonical_data.py`。
训练时把接管段和原始示教数据放在一起（DAgger 聚合）。

## 4. 测试

```bash
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
conda activate aloha
pip install pytest        # 只有跑测试才需要
python3 -m pytest aloha_scripts/tests -q
```

覆盖：msgpack 与 openpi_client 互通、resize 与 openpi 逐像素一致、chunk 执行（逐步回放、30 Hz 插值、出错转暂停）、
离合主臂（增益渐降、断扭矩顺序、稳定检测、offset、锁定顺序）、从臂限位/限速/缓入、状态机全流程、两种格式的保存与切分，
以及用假硬件跑通 `main()`（openpi + hdf5、N0 + VTLA）。触觉相关（`test_tactile_pipeline.py`、
`test_tactile_recording_flows.py`）：raw / jpeg 解码、按数据年龄判断过期、清零与 runtime config、时间戳和
`check_episode_timing.py`，以及用假硬件和模拟触觉帧跑通三个录制入口（`record_episodes_aloha2.py`、
`record_episodes_aloha2_vtla.py`、`hil_rollout.py` 的 hdf5 + 触觉），包括两个录制脚本中途断流后自动重录。
