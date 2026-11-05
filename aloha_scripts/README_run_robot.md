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

## 4. 用 HIL 数据继续训练

`hil_split_episodes.py` 默认 `--keep intervention`：只留"人类真正在控制手臂"的那些段，当成 DAgger 要喂给模型的纠正数据。
**不要**用 `--keep active`/`autonomous` 来做训练数据——`active` 会把策略当时犯错的动作也当成"正确答案"喂进去，`autonomous` 等于让模型学它自己已经会的东西，都没有新信息；这两个选项更适合调试/看效果，不是用来训练的。

> 补充：Berkeley 的 DAgger 实践（见下面"demo : HIL 配比怎么定"）发现**纯用 intervention 不一定最稳**——他们额外保留了新一轮里策略自己正常跑、没被打断的那段（`--keep autonomous`，但是**这一轮新录的**，不是旧的 autonomous 数据），和接管数据按差不多的量级混进去，效果比纯 intervention 或纯整条 episode 都好。如果纯 `--keep intervention` 训出来效果不稳，这是一个可以试的方向，不是必须改。

切出来的 episode（`<task>_hil_corrections/episode_K.hdf5` 或 `episode_K/`）和普通录制**完全同格式**，后面两条训练路径走法不一样：

### demo : HIL 配比怎么定

没有统一的"标准比例"，但两条真实案例给的数字比空想靠谱：

- **Berkeley 团队**（论文《Scalable Behavior Cloning with Open Data, Training, and Evaluation》，代码见 [i2rt-robotics/abc](https://github.com/i2rt-robotics/abc)）的 DAgger 配比：**80% 上一轮数据 + 10% 本轮接管纠正 + 10% 本轮剩下的部分**（同一批新 episode 里策略自己正常跑、没被打断的那段）。他们测试过"只用接管数据"和"只用完整 episode"两种极端，**都比这个配比差**——纯接管数据会让模型在"旧 demo 场景"和"新纠正场景"之间学出虚假关联。
- **反面案例**（Pebblous 博客，真实 GR00T 训练记录）：加入 93 条纠正 episode（占混合数据 24%）**不加权直接拼接**，闭环成功率从 73% 掉到 43%，offline 指标还看不出问题。**不加权拼接不只是把纠正信号稀释，甚至可能主动把模型带坏**。
- **正面案例**（Axis Robotics 博客）：大约 **2:1（原始 : 新纠正数据）** 混合，闭环成功率 57.1% → 66.3%。

**"不加权直接拼"具体什么意思**：LeRobot 的 `MultiLeRobotDataset`（openpi 合并数据集用的类）`__len__()` 返回总帧数，`__getitem__(idx)` 按全局帧序号直接取——相当于把两份数据的所有帧摊成一个大列表，`DataLoader` 打乱后均匀抽，每一帧中标概率都一样，**代码里压根没有调这个概率的机制**，不是故意调低。实测 `packaging_simple_2_hil` 的 7 条 HIL episode：有接管的 5 条平均每条真正 intervention 的帧数只有约 500 帧（10 秒左右），另外 2 条接管帧数是 0（录了但没真纠正，切完贡献不了数据）；而完整 demo 一条约 2500~3800 帧。50 条 demo + 10 条 HIL 直接合并，纠正帧数大概只占总帧数的 3%~4%，几乎被淹没。

两条训练路径因此做法不同：

- **ACT**：用现成的 `sample_weights`（见下面 ACT 小节），直接按权重控制占比，不用动文件。
- **openpi / pi0.5**：没有数据集级别的加权字段，想把 HIL 占比抬到 10%~20%（对应 Berkeley 的量级），只能用物理手段——把切出来的 HIL 纠正 episode 在合并前复制 3~5 份再跑转换脚本，粗暴但有效。合并完**务必做闭环 eval**，不能只看 loss/offline 指标（Pebblous 的案例就是 offline 看不出问题，实际部署才发现变差）。

### ACT（本地，这个仓库自己的 `imitate_episodes_multi_gpu.py`）

`hil_split_episodes.py` 切出来的 hdf5 可以直接喂给 ACT，不需要任何转换——格式和 `record_episodes_aloha2.py` 的原始录制完全一样，`utils.py` 的 `find_all_hdf5()` 只是递归找所有 `*.hdf5`，不关心文件名和目录结构。

把纠正数据和原始示教数据合在一起训练，推荐在 `constants.py` 里仿照已有的 `zip_tie_cotrain` 新建一个 cotrain 任务配置：

```python
'packaging_cotrain': {
    'dataset_dir': [DATA_DIR + '/packaging', DATA_DIR + '/packaging_hil_corrections'],
    'sample_weights': [1, 3],   # 可选：纠正数据通常量少但更珍贵，适当加权重采样
    'episode_len': 1600,
    'camera_names': [...],      # 和原任务一致
},
```

然后照常训练，用 `--resume_ckpt_path` 从旧 checkpoint 热启动（README.md"Policy Training"小节本来就是给这个场景写的）：

```bash
python imitate_episodes_multi_gpu.py \
    --task_name packaging_cotrain \
    --ckpt_dir <新的 ckpt_dir> \
    --policy_class ACT \
    ...（其它超参和之前一致）\
    --resume_ckpt_path <旧 ckpt_dir>/policy_last.ckpt
```

`--resume_ckpt_path` 只加载模型权重，不恢复 optimizer 状态和 step 计数，所以 `--num_steps` 填的是"这次新训练要跑多少步"。`dataset_stats.pkl`（归一化用的均值方差）每次训练都会在 `--ckpt_dir` 里重新算一份，eval 时也是读新 ckpt_dir 自己的 stats，新旧 checkpoint 之间不会因为数据集变了对不上，不用担心。练完之后照常 `--eval --ckpt_dir <新 ckpt_dir>` 部署，没有额外转换步骤。

### openpi / pi0.5（服务器端）

openpi 不直接吃原始 hdf5，训练用的是 LeRobot 格式数据集（`serve_policy.py` 部署时读的 `norm_stats.json` 就在这个数据集目录下）。以下流程已经在服务器上对着真实脚本和历史 log（`packaging_simple_2_plus_pipeline.log`）核对过：

1. **转换 + 合并进 LeRobot 数据集**（openpi 仓库自带的转换脚本支持往已有数据集上追加新 episode，`--base-repo-id` 指向的旧数据集本身不会被改动，只是硬链接，不占新磁盘空间）：

   ```bash
   cd /mnt/ssd2/guningquan/Programs_server/openpi
   uv run examples/aloha_real/convert_aloha_data_to_lerobot_no_bgr2rgb.py \
     --raw-dir <task>_hil_corrections \
     --base-repo-id /mnt/ssd2/guningquan/Dataset/packaging_simple_2_lerobot \
     --repo-id /mnt/ssd2/guningquan/Dataset/packaging_simple_2_hil_lerobot \
     --task "packaging the items" --num-workers 6
   ```

   `--raw-dir` 要求目录里直接是 `episode_N.hdf5`（不递归找子文件夹），`hil_split_episodes.py` 切出来的文件名正好是这个格式。
2. **在 `src/openpi/training/config.py` 里新加一个 `TrainConfig`**：复制现有的 `pi05_packaging_simple_2_plus_full` 那段，只改 `data.repo_id` 指向上一步的新数据集。`weight_loader`/`pytorch_weight_path` 从 0 开始还是接着练，两种都可以：
   - **从 0 开始**：保持指向 pi0.5 官方基础模型（`gs://openpi-assets/checkpoints/pi05_base/params` / `pytorch_weight_path=.../pi05_base_pytorch`）——这个项目里 `pi05_packaging_simple_2_full`/`_plus_full` 这几个配置目前都是这么写的，但这是**这几个 config 自己的选择，不是框架限制**。
   - **接着上一次任务 checkpoint 继续练**：把 `pytorch_weight_path` 改指向旧 checkpoint 的 step 目录，比如 `.../pi05_packaging_simple_2_full/packaging_simple_2_pt/20000`。`scripts/train_pytorch.py` 存 checkpoint 时写的就是一个 `model.safetensors`（第 167 行），加载 `pytorch_weight_path` 时读的也是同名文件（第 445-446 行），格式完全兼容，不用改代码。`config.py` 里也有先例：`pi05_droid_finetune` 的 `weight_loader` 就指向另一个已微调过的 checkpoint（`gs://openpi-assets/checkpoints/pi05_droid/params`），不是最原始的 base model。
     这种方式是**只加载权重**（跟 ACT 的 `--resume_ckpt_path` 同一类），不等于 `--resume`——`--resume` 是接着同一个 `exp_name` 的中断训练，连 optimizer 状态和 step 数一起恢复，针对的是训练中断重跑，不是"接上一个任务的 checkpoint 开始新任务"。
3. **算归一化统计**（合并数据集后必须重新算，不会自动生成）：

   ```bash
   uv run scripts/compute_norm_stats.py <新 config 名>
   ```
4. **训练**（多卡用 `torchrun`）：

   ```bash
   torchrun --standalone --nnodes=1 --nproc_per_node=<GPU 数> scripts/train_pytorch.py <新 config 名> --exp_name=<新实验名>
   ```
5. **部署**：和现在一样，`serve_policy.py --policy.config=<新 config 名> --policy.dir=<checkpoint 目录>/<新实验名>/<step>`。

这一整套（采集 HIL → 切纠正段 → 合并进训练集 → 续练/重新微调 → 换新 checkpoint 部署 → 继续采集）就是标准 DAgger 循环。

---

## 5. 常用参数（`hil_rollout.py`；`run_pi_policy.py` 的参数见第 2 节和 `--help`）

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

## 6. 常见问题

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
