# 本地启动机器人并连接服务器策略（正常执行 / HIL）

机器人端（本机，ROS noetic + conda 环境 `aloha`，Python 3.8）只负责采集观测、执行动作、接管和录制；模型在 GPU 服务器上，
由 openpi 或 N0-VTLA 仓库自己的 `scripts/serve_policy.py` 加载，两者都是 websocket + msgpack，默认端口 8000。

```
[服务器 10.240.77.20]  serve_policy.py  ◀── websocket :8000 ──▶  [本机] run_pi_policy.py / run_vtla_policy.py / hil_rollout.py ── ROS ── 手臂 + 相机 (+ 触觉)
```

| 场景                                            | 本机脚本               |
| ----------------------------------------------- | ---------------------- |
| 正常执行 openpi（纯推理验证，不接管、不存数据） | `run_pi_policy.py`   |
| 正常执行 N0-VTLA（同上）                        | `run_vtla_policy.py` |
| HIL 采集（人工接管 + 录制）                     | `hil_rollout.py`     |

HIL 的设计、数据格式、切分见 `README_hil.md`；本文件只讲怎么启动、怎么按键。

---

## 0. 环境

本机脚本都在 conda 环境 `aloha` 里运行（已带 msgpack、websockets、h5py_cache、modern_robotics、evdev，不用另装）：

```bash
conda activate aloha
```

`constants.py` 的 `DATA_DIR` 按登录用户名选路径（`ubuntu20` → `/home/robot/Dataset_and_Checkpoint/dataset`）。

## 1. 启动顺序（三个终端）

### 终端 1（本机）：ROS

手臂上电、四条臂处在休息位后：

```bash
# 只用相机（openpi pi0.5 等）：
roslaunch aloha 4arms_teleop.launch
# 要触觉（N0-VTLA 必须）或主臂夹爪力反馈：
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:="force"
```

这个终端一直开着。没起触觉时，力反馈订阅不到数据只会警告（主臂夹爪保持 0 PWM），也可以加 `--no_force_feedback`。

### 终端 2（服务器）：起推理服务

```bash
ssh server
nvidia-smi                                   # 找一张空闲的卡
```

openpi（pi0.5 packaging，归一化统计按配置里的绝对路径自动读取，不用拷）：

```bash
cd /mnt/ssd2/guningquan/Programs_server/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_packaging_simple_2_full \
  --policy.dir=/mnt/ssd2/guningquan/Programs_server/openpi_checkpoints/pi05_packaging_simple_2_full/packaging_simple_2_pt/20000
```

N0-VTLA（`--policy.config` 与 checkpoint 训练时一致：joint 用 `vtla_joint_posttrain`，eef 用 `vtla_tactile_posttrain`）：

```bash
cd /mnt/ssd2/guningquan/Programs_server/N0-VTLA
CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port 8000 \
  --policy.config=vtla_joint_posttrain --policy.dir=checkpoints/vtla_joint_posttrain/<exp>/<step>
```

看到服务开始监听（`Creating server ...`）即可。连不上 8000 端口时用 SSH 隧道，本机命令里改成 `--host 127.0.0.1`：

```bash
ssh -N -L 8000:localhost:8000 server         # 本机再开一个终端执行，保持不关
```

### 终端 3（本机）：机器人端

```bash
conda activate aloha
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python3 aloha_scripts/run_pi_policy.py ...   # 正常执行，见第 2 节
python3 aloha_scripts/hil_rollout.py ...     # HIL，见第 3 节
```

**按键只在这个终端里生效**（要保持焦点在这个终端；模拟键盘方向键的脚踏板同理）。

用 `--pedal_device` 接 evdev 脚踏板时，脚本会独占这个设备（踏板按键不再同时敲进终端），三个踏板分别等于
→（`--pedal_pause`，默认 KEY_A）、←（`--pedal_correct`，KEY_B）、e（`--pedal_end`，KEY_C）；
开始和保存的提示里也能用：暂停踏板 = 开始 / 保存，接管踏板 = 丢弃。每条开始时会清空之前积压的按键和踏板事件。

`hil_rollout.py` 启动后依次做（`run_pi_policy.py` 只动从臂，见第 2 节）：

1. 连接服务器（最多等 30 s）。
2. **四条臂移动到起始位**（`opening_ceremony`），手离开机械臂。
3. 有力反馈时标定主臂夹爪零点：这时不要碰主臂夹爪。
4. **预热推理**：只在第一条之前做一次。openpi 第一次推理要编译模型，可能几分钟，期间机器人不动；最长等 `--warmup_timeout`（900 s）。
5. 打印 `Ready. RIGHT = start the policy`，按 → 开始。

全部结束（或 ESC）后，四条臂回休息位并断扭矩。Ctrl+C 也会走这一步。

---

## 2. 正常执行（纯推理验证）

只跑模型、不接管、不存数据，主臂不参与。openpi 用 `run_pi_policy.py`，N0 用 `run_vtla_policy.py`，两个脚本按键一致。

### openpi：`run_pi_policy.py`

```bash
python3 aloha_scripts/run_pi_policy.py --host 10.240.77.20 --port 8000 --prompt "packaging the items" --num_episodes 10
```

观测和执行方式与 openpi 自带的 `examples/aloha_real/main.py` 相同（qpos + 三路相机 resize 到 224，每 50 步阻塞推理一次，50 Hz），
另外加了从臂限位 / 每 tick 限速（`--max_joint_step`）、相机过期检查（`--max_stream_age`）和首次预热推理。

启动后：从臂回起始位（夹爪张开）→ 预热推理（只做一次，openpi 首次推理要编译，可能几分钟）→ 提示按 → 开始。

| 键       | 作用                                                   |
| -------- | ------------------------------------------------------ |
| →       | 开始本轮                                               |
| ← 或 r  | 中止本轮：从臂停在原地，然后回起始位，等 → 开始下一轮 |
| ESC 或 q | 停止：夹爪张开，从臂回休息位并断扭矩                   |

- 推理出错 / 超时（`--infer_timeout` 10 s）或相机断流时，本轮按 ← 的方式中止。
- 每轮最长 `--max_episode_steps`（3000 tick = 60 s），跑满算完成一轮；中止的不计入 `--num_episodes`。
- 每次推理执行 `--action_horizon`（50）步。

### N0-VTLA：`run_vtla_policy.py`

```bash
python3 aloha_scripts/run_vtla_policy.py --host 10.240.77.20 --port 8000 --action_space joint --prompt "Perform the task."
```

按键同上（→ 开始、← 或 r 中止并回起始位、ESC 或 q 停止）。另有 `--debug` 模式：每按一次 Enter 采一帧、显示图像
（窗口里 q 下一张、ESC 跳过）、推理并打印动作形状，不动手臂；输入 q 退出。

---

## 3. HIL 采集

```bash
# openpi，存 hdf5 到 /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil/episode_N.hdf5
python3 aloha_scripts/hil_rollout.py --task_name packaging_simple_2 --policy openpi \
  --host 10.240.77.20 --port 8000 --prompt "packaging the items" --num_episodes 20

# N0-VTLA，存 VTLA 原始目录 <dataset_dir>_hil/episode_N/
python3 aloha_scripts/hil_rollout.py --task_name aloha2_tactile_test --policy n0_joint \
  --host 10.240.77.20 --prompt "Perform the task." --num_episodes 20
```

`--output_dir` 可改保存目录；`--num_episodes` 是要**保存**的条数（丢弃的不算）。

每条的长度上限 `--max_timesteps`（默认任务的 `episode_len`，packaging_simple_2 为 2500 tick = 50 s）只计算**策略执行 + 人工接管**的时间，暂停、对齐、主臂变软的时间不算；到上限时如果正在接管，会提示 `Time limit reached: hand back (LEFT)`，等你按 ← 交还后再结束。含暂停在内的总时长最多是上限的 4 倍。

运行中只有一行进度条原地刷新：`有效 tick / 上限 [用时, 当前阶段 | 已录帧数 | 接管次数 | 最近一次推理耗时 | 接管时的 offset]`，暂停、接管等事件信息打印在进度条上方。

按 ← 之后到接管开始之前（仍是黄色 `paused`），这一行还会显示准备进度（两只主臂里更慢的那只）：先是 `releasing [██████░░░░] 7/12`（主臂变软，还不用动手），
再是 `settling [████░░░░░░] steady 0.20/0.50s | move 0.014/0.010 rad (R elbow) | wait 0.6/2.0s`（把主臂放稳）。
进度条随"连续稳定时间"涨到 `--settle_s`，某个关节变化超过 `--settle_tolerance` 就清零并标出是哪个关节；涨满后整行变红，从臂开始跟随你。
`wait` 到 `--settle_timeout_s` 也会直接开始接管。变红之前你动的部分不会传给从臂，所以先稳住再操作。这段颜色一直是黄色系：绿＝策略在跑，红＝你在接管。详见 `README_hil.md`。

### 按键

| 键         | 什么时候   | 作用                                                                                     |
| ---------- | ---------- | ---------------------------------------------------------------------------------------- |
| →         | 等待开始时 | 开始本条，策略接管从臂                                                                   |
| →         | 策略执行中 | **暂停**：从臂原地保持，主臂在 2 s 内自动移到从臂的位置                            |
| ←         | 暂停中     | **接管**：主臂变软（0.5 s）→ 断扭矩 → 等主臂停稳（≤2 s）→ 从臂开始跟随你的主臂 |
| ←         | 接管中     | **交还**：主臂锁住，回到暂停                                                       |
| →         | 暂停中     | **恢复策略**                                                                       |
| e 或 Enter | 任意时刻   | 结束本条 → 询问保存                                                                     |
| ESC 或 q   | 任意时刻   | 结束本条 → 询问保存 → 退出                                                             |
| b          | 非接管时   | 重新标定主臂夹爪力反馈零点（标定会让控制循环停一下，所以接管中不响应）                   |

结束本条后的询问（从臂夹爪会先张开，夹着东西会掉）：

| 键  | 作用                                                   |
| --- | ------------------------------------------------------ |
| →  | 保存                                                   |
| ←  | 丢弃（同一编号重录）                                   |
| ESC | 丢弃并退出（如果是按 ESC 结束的，则 →/← 选完就退出） |

无效按键不会执行：执行中按 ← 提示先暂停；接管中按 → 提示先按 ← 交还；主臂还在变软/稳定时按 → 提示稍等、再按 ← 不起作用。

### 一次典型的接管

1. 策略执行中看到要出错 → 按 **→** 暂停。从臂停住，主臂自动对齐到从臂（约 2 s，别硬掰主臂）。
2. 手握住两个主臂把手 → 按 **←**。主臂逐渐变软（不会突然掉），停稳后终端打印
   `relative control from here ... largest offset ...`，从臂开始跟手。主臂松手时的小幅下坠已被离合 offset 抵消，从臂不会跳。
   offset 超过 0.25 rad 会警告：说明主从姿态差得多，最好交还后重新暂停对齐。
3. 纠正完 → 按 **←** 交还（主臂锁住）→ 按 **→** 让策略继续。可以多次重复。
4. 任务完成 → 按 **e**，再按 **→** 保存。

每次接管的前 0.2 s（`--correction_skip_s`）照常录制但不标为 intervention。默认不缓入（`--correction_blend_s 0`）：离合模式下接管第一条指令就等于从臂保持的指令，本来就连续。这些默认值与你的 lerobot 命令一致。

### 采完之后

```bash
# 只要接管段（DAgger 数据），再用原来的转换脚本转成 LeRobot 格式训练
python3 aloha_scripts/hil_split_episodes.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections --keep intervention
```

### 转成视频看一眼

转换/训练之前，先把某一条 `episode_N.hdf5` 转成带标注的视频，直观看一下哪些是机器自己跑的、哪些是人接管的：

```bash
python3 aloha_scripts/hil_episode_to_video.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil/episode_0.hdf5 \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil/episode_0_annotated.mp4
```

每个摄像头画面并排拼接，边框颜色标出当前状态（终端也会打印同样的统计和分段时间表）：

- 绿色 `AUTONOMOUS` — 策略自己在跑（`hil/phase == 0`）
- 黄色 `PAUSED` — 策略暂停，等待决策（`hil/phase == 1`）
- 深红 `CORRECTING` — 已经进入接管流程，但还在前 `--correction_skip_s` 内，暂不计入 intervention（`hil/phase == 2` 且 `hil/intervention == 0`）
- 亮红 `HUMAN` — 人类正在实际接管，真正算作 intervention 的那部分（`hil/intervention == 1`）

`--fps` 可指定输出帧率（默认 50，即 `hil_rollout.py` 的采集频率）。普通（非 HIL）的 `record_episodes_aloha2.py` 录制也能喂给这个脚本，只是整段都会标成 `AUTONOMOUS`。

---

## 4. 常用参数（`hil_rollout.py`；`run_pi_policy.py` 的参数见第 2 节和 `--help`）

| 参数                                               | 默认      | 说明                                                                            |
| -------------------------------------------------- | --------- | ------------------------------------------------------------------------------- |
| `--action_horizon`                               | 50        | 每次推理执行多少步                                                              |
| `--infer_timeout`                                | 10        | 单次推理超时（秒），超时自动暂停                                                |
| `--align_s`                                      | 2.0       | 暂停时主臂对齐到从臂的时间                                                      |
| `--release_ramp_s`                               | 0.5       | 接管时主臂变软的时间，觉得太快就调大                                            |
| `--settle_s` / `--settle_timeout_s`            | 0.5 / 2.0 | 主臂停稳判定 / 最长等待                                                         |
| `--absolute_takeover`                            | 关        | 不用离合，从臂直接跟主臂绝对位置                                                |
| `--correction_blend_s` / `--correction_skip_s` | 0 / 0.2   | 接管开始的缓入（从上一条指令开始）/ 不标记时长                                  |
| `--max_joint_step`                               | 0.1       | 从臂每 tick 关节最大变化（rad），0 关闭                                         |
| `--max_stream_age`                               | 0.5       | 相机数据超过这个秒数就暂停策略                                                  |
| `--no_force_feedback`                            | 关        | 关闭主臂夹爪力反馈                                                              |
| `--pedal_device`                                 | 无        | evdev 脚踏板（`--pedal_pause KEY_A --pedal_correct KEY_B --pedal_end KEY_C`） |

`python3 aloha_scripts/hil_rollout.py --help` 看全部参数。

## 5. 常见问题

| 现象                                                    | 处理                                                                                                      |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `Waiting for policy server ...` 一直重试              | 服务没起来或端口不通：服务器上确认在监听；用 SSH 隧道 + `--host 127.0.0.1`                              |
| `ModuleNotFoundError: msgpack` 等                     | 没激活环境：`conda activate aloha`（系统 Python 没有这些包）                                            |
| `Warming up the policy server` 很久                   | openpi 首次推理在编译，正常；超过 `--warmup_timeout` 才报错                                             |
| 执行中自动 `Paused (inference failed / no reply ...)` | 服务端报错或超时：看服务器终端；GPU 被占满时推理变慢，可调大 `--infer_timeout`                          |
| 自动 `Paused (stale stream ...)`                      | 某路相机/触觉断流：检查终端 1 的 launch 输出、`rostopic hz /usb_cam_high/image_raw`                     |
| `needs a terminal (stdin is not a tty)`               | 必须在交互终端里直接运行，不能放后台或重定向 stdin                                                        |
| 从臂动作一顿一顿                                        | 每个 chunk 推理时会停顿一次（与 openpi aloha_real 相同）；推理慢时看服务器 GPU 是否被占                   |
| `follower: command jumped ... limited`                | 策略输出或主臂读数单步跳变被限速，偶发可忽略，频繁出现要查原因                                            |
| `leader Position_P_Gain is [...]` 启动报错            | 上次运行在主臂变软的过程中被强杀，舵机增益（舵机 RAM 寄存器）停在低值；主臂断电重启、再重启 launch 后再跑 |
| 推理超时后按键一段时间没反应                            | 推理是阻塞的，超时要等 `--infer_timeout`，恢复策略时还要重连服务器；这期间从臂保持不动                  |
