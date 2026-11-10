# 本地启动机器人并连接服务器策略（正常执行 / HIL）

机器人端（本机，ROS noetic + conda 环境 `aloha`，Python 3.8）只负责采集观测、执行动作、接管和录制；模型在 GPU 服务器上，
由 openpi 或 N0-VTLA 仓库自己的 `scripts/serve_policy.py` 加载，两者都是 websocket + msgpack，默认端口 8000。

```
[服务器 00010.240.77.20]  serve_policy.py  ◀── websocket :8000 ──▶  [本机] run_pi_policy.py / run_vtla_policy.py / hil_rollout.py ── ROS ── 手臂 + 相机 (+ 触觉)
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
触觉 launch 启动时会给传感器清零，启动时不要碰传感器；`hil_rollout.py` 每条 episode 开始前也会清零一次
（`--no_tactile_calibration` 关闭），触觉帧本身超过 `--max_tactile_age`（默认 0.2 s）时暂停策略。

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

LeRobot 训练的模型（pi0.5、MolmoAct2、ACT、GR00T 等，要求是用本仓库 hdf5 转换的数据训练的）用 lerobot 仓库的
`examples/aloha_ros/serve_policy_websocket.py`，协议和 openpi 相同，机器人端加 `--policy lerobot`。运行环境是服务器上
的 conda 环境 `lerobot`（已经装好 `websockets`、`msgpack`）：

```bash
conda activate lerobot                        # /mnt/ssd1/guningquan/miniforge3/envs/lerobot
cd /mnt/ssd2/guningquan/Programs_server/lerobot
CUDA_VISIBLE_DEVICES=0 python examples/aloha_ros/serve_policy_websocket.py \
  --policy_path <checkpoints/.../pretrained_model> --task "packaging the items" --port 8000
```

GR00T（例如 `outputs/picking_w_cover`）还要先装依赖（只需一次），命令同上，换 `--policy_path` 和 `--task`：

```bash
pip install "dm-tree>=0.1.8,<1.0.0" "timm>=1.0.0,<1.1.0" "decord>=0.6.0,<1.0.0" "diffusers>=0.38.0,<0.40.0"
CUDA_VISIBLE_DEVICES=2 python examples/aloha_ros/serve_policy_websocket.py \
  --policy_path /mnt/ssd2/guningquan/Programs_server/lerobot/outputs/picking_w_cover \
  --task "picking ichigo to the cup" --port 8000
```

机器人端对应的 `run_pi_policy.py` / `hil_rollout.py` 命令见 lerobot 仓库 `examples/aloha_ros/README_CN.md` 9.4 节。

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

只跑模型、不接管、不存数据，主臂不参与。openpi 和 LeRobot 模型用 `run_pi_policy.py`，N0 用 `run_vtla_policy.py`，两个脚本按键一致。

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

### LeRobot 模型：`run_pi_policy.py --policy lerobot`

```bash
python3 aloha_scripts/run_pi_policy.py --policy lerobot --host 10.240.77.20 --port 8000 --num_episodes 10
```

服务器是 lerobot 的 `serve_policy_websocket.py`（见第 1 节终端 2）。按键、执行方式、安全限制都和上面的 openpi 一样，区别只在发送的观测：
qpos、qvel、effort，加上 checkpoint 用到的相机（连接时由服务器告知，`--cameras` 可改），原始分辨率、JPEG 质量 90（和录制时一样，
`--jpeg_quality` 可改），每次约 150 kB。`--prompt` 不给时用服务器的 `--task`。每次推理执行 `--action_horizon`（50）步，
不超过模型返回的步数（比如 MolmoAct2 是 30）。

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

# LeRobot 模型（服务器跑 serve_policy_websocket.py），同样存 hdf5
python3 aloha_scripts/hil_rollout.py --task_name packaging_simple_2 --policy lerobot \
  --host 10.240.77.20 --port 8000 --num_episodes 20

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

**"不加权直接拼"具体什么意思**：LeRobot 的 `MultiLeRobotDataset`（openpi 合并数据集用的类）`__len__()` 返回总帧数，`__getitem__(idx)` 按全局帧序号直接取——相当于把两份数据的所有帧摊成一个大列表，`DataLoader` 打乱后均匀抽，每一帧中标概率都一样，**代码里压根没有调这个概率的机制**，不是故意调低。实测（用下一节的 `hil_compare_lengths.py`）：`packaging_simple_2` 的 50 条 demo 共 91,616 帧（平均每条约 1,830 帧，最长 2,500）；`packaging_simple_2_hil` 的 7 条 HIL episode 共 21,322 帧，其中真正接管的只有 2,489 帧（5 条各约 500 帧，10 秒左右），另外 2 条接管帧数是 0（录了但没真纠正，切完贡献不了数据），还有约 30%（6,487 帧）是暂停、从臂不动。接管数据只占原数据的 2.7%，直接合并几乎被淹没。

两条训练路径因此做法不同：

- **ACT**：用现成的 `sample_weights`（见下面 ACT 小节），直接按权重控制占比，不用动文件。
- **openpi / pi0.5**：没有数据集级别的加权字段，想让新增的 HIL 数据占原数据的 10%~20%（对应 Berkeley 的量级），要么多采，要么用物理手段——把切出来的 HIL 纠正 episode 在转换前复制若干份（见下面 openpi 小节的符号链接写法），粗暴但有效。合并完**务必做闭环 eval**，不能只看 loss/offline 指标（Pebblous 的案例就是 offline 看不出问题，实际部署才发现变差）。

换算成"新增数据占原数据的百分比"（new/base）：Berkeley 的 10% 接管对 80% 旧数据约是 12.5%，Axis 的 2:1 是 50%。所以 new/base 取 10%~20%，和 Berkeley 同一量级，比 Axis 保守。

### 先量长度，再定配比：`hil_compare_lengths.py`

不用手算，直接量两份数据的长度，看新增的接管数据占原数据百分之几、要复制几份或再多采多少才到 10% / 20%：

```bash
python3 aloha_scripts/hil_compare_lengths.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil
# 默认只数新数据里"人真正接管"的部分（≥ --min_frames 25 帧的段，口径和 hil_split_episodes.py --keep intervention 完全一致）
# --ratio 0.1 0.2   想要的 新增/原数据（默认 10% 和 20%）
# --part all        新数据整条都数；--part autonomous|active 对应另外两种切法
# --per_episode     每条的帧数和段数
```

第一个参数是原数据：`episode_*.hdf5` 目录，或已经转好的 LeRobot 数据集（读 `meta/info.json` 的 `total_frames`）。第二个参数是 `hil_rollout.py` 录的目录（切分前后都行）。帧数按 `/action` 数，hdf5 为 50 Hz。对上面这两个目录的输出：

```text
base : .../packaging_simple_2  [hdf5]  50 episodes, 91,616 frames = 30.5 min @50 Hz (avg 1,832/episode)
new  : .../packaging_simple_2_hil  [hdf5]  7 episodes, 21,322 frames = 7.1 min
       autonomous 12,281 | paused 6,487 | correcting-skip 65 | intervention 2,489
       counted: --part intervention (segments >= 25 frames) = 5 segment(s) in 5 of 7 episodes = 2,489 frames = 49.8 s

new / base = 2.7%    share of new in base + new = 2.6%
  new/base  10%  ->  k = 4  copies (new/base 10.9%, share of merged 9.8%, 9,956 frames); or record 6,673 more frames (~14 more corrections of 498)
  new/base  20%  ->  k = 7  copies (new/base 19.0%, share of merged 16.0%, 17,423 frames); or record 15,834 more frames (~32 more corrections of 498)
```

读法：现在只有 2.7%，到 10% 要把这 5 段复制 4 份，或者再多接管约 14 次；到 20% 要复制 7 份，同样的 5 段被重复 7 遍，不如多采。复制份数超过 5 时工具会提示"重复太多"。

### ACT（本地，这个仓库自己的 `imitate_episodes_multi_gpu.py`）

`hil_split_episodes.py` 切出来的 hdf5 可以直接喂给 ACT，不需要任何转换——格式和 `record_episodes_aloha2.py` 的原始录制完全一样，`utils.py` 的 `find_all_hdf5()` 只是递归找所有 `*.hdf5`，不关心文件名和目录结构。

把纠正数据和原始示教数据合在一起训练，推荐在 `constants.py` 里仿照已有的 `zip_tie_cotrain` 新建一个 cotrain 任务配置：

```python
'packaging_cotrain': {
    'dataset_dir': [DATA_DIR + '/packaging', DATA_DIR + '/packaging_hil_corrections'],  # 只有第一个目录参与验证集划分
    'sample_weights': [90, 10],  # 每个 batch 里抽自各目录的概率（归一化后），不是长度比例
    'episode_len': 1600,
    'camera_names': [...],       # 和原任务一致
},
```

**`sample_weights` 是"每个目录被抽到的概率"，和目录里有多少帧无关**：`utils.py` 的 `BatchSampler` 先按权重选目录，再在该目录里均匀抽一帧。所以：

- **不写 `sample_weights` 就是各目录 50/50**——纠正数据只有几千帧，也会占一半的 batch，同样的帧被反复看，很容易过拟合。
- `[1, 3]` 是 25% / 75%，不是"1:3 的数据量比例"。
- 想让接管数据占 10% 就写 `[90, 10]`（`hil_compare_lengths.py` 的输出里有对应的 p；新增/原数据 10% 对应 p ≈ 9%，20% 对应 p ≈ 17%）。每个纠正帧在整个训练里大约被看到 `步数 × batch_size × p ÷ 纠正总帧数` 次，纠正帧很少时这个次数会很大，要关注。
- 归一化统计默认用 `dataset_dir` 里所有目录合并后重算；想和旧 checkpoint 保持一致，可以像 `zip_tie_cotrain` 那样加 `'stats_dir': [DATA_DIR + '/packaging']`，只用原数据算。

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

### LeRobot v3.0（lerobot 仓库训练 pi05 / MolmoAct2 / ACT）

lerobot 仓库训练用的是 LeRobot v3.0 数据集（和 openpi 用的 v2.1 不是同一个版本）。下面两个脚本都在本目录，但要 import lerobot，所以用 **lerobot 的 venv**（Python 3.12）运行，不是 conda 环境 `aloha`：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
```

**hdf5（含 `hil_rollout.py` 切出来的纠正段）→ v3.0**：`--raw-dir` 可以给多个目录，按顺序合成一个数据集；同一个目录写两次就转两份（相当于复制 k 份抬高占比）。动作按录制时的绝对值存，`meta/stats.json` 是全部数据的精确分位数，`meta/source_episodes.tsv` 记每条 episode 的来源文件。

```bash
python aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py \
  --raw-dir /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
            /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_v30 \
  --task "packaging the items" --fps 50
```

多份数据要一起转，不要分开转完再用 `lerobot-edit-dataset merge` 合并：LeRobot 的合并和重算统计都只保留每条 episode 分位数的包络（q01 取各 episode 里最小的 q01），范围偏宽。

**`lerobot-rollout --strategy.type=dagger` 录的 HIL → 并进已有的 v3.0 数据集**：dagger 只写它自己的 `rollout_*` 数据集（每次纠正一条 episode），不会动训练用的数据集。合并输出一个新数据集，输入都不改：

```bash
python aloha_scripts/merge_hil_into_lerobot_v30.py \
  --base /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 \
  --hil /home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30_hil
```

两种数据集的特征不同，脚本会先对齐：去掉 HIL 的 `intervention`（纠正模式下全是 True）和 `cam_low`，去掉 base 的 velocity/effort（dagger 不录；pi05、MolmoAct2、ACT 只用 `observation.state`），`robot_type` 统一成 base 的。视频直接拼接不重编码，统计重算成精确分位数，最后打印 HIL 帧的占比。HIL 必须是加了 `--robot.gripper_state_format=aloha2` 录的。

**两种 HIL 一起用**：先用转换脚本把示教和 `hil_rollout.py` 的纠正段一起转成 v3.0，再用合并脚本把 dagger 数据集并上去。顺序不能反（hdf5 只能走转换脚本，LeRobot 数据集只能走合并脚本）；dagger 的数据也不能反过来给本仓库的 ACT 或 openpi 用。要达到 `hil_compare_lengths.py` 算出的 k 份，就把纠正段目录在 `--raw-dir` 里写 k 次。两种方式的对比和完整流程见 lerobot 仓库的 `examples/aloha_ros/README_CN.md` 第 10 节。

之后在 lerobot 仓库里，用 `lerobot-train --policy.path=<旧 pretrained_model> --dataset.root=<新数据集> ...` 接着旧模型练（归一化统计自动换成新数据集的，不要用 `--resume=true`），完整命令见 lerobot 仓库的 `examples/aloha_ros/README_CN.md` 第 8、10 节。

### openpi / pi0.5（服务器端）

openpi 不直接吃原始 hdf5，训练用的是 LeRobot 格式数据集（`serve_policy.py` 部署时读的 `norm_stats.json` 就在这个数据集目录下）。以下流程已经在服务器上对着真实脚本和历史 log（`packaging_simple_2_plus_pipeline.log`）核对过：

**并进去有两种做法，推荐第一种**（以下说的"新 episode"就是切出来的接管段，复制份数见上一节 `hil_compare_lengths.py`）：

|            | A. 只转新的，追加到已转好的 LeRobot 数据集（`--base-repo-id`，推荐）                                                                                                   | B. 两类 hdf5 用符号链接放进同一个目录，整体重转                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| 做法       | 只转换新 episode；旧数据集的 parquet 用硬链接带过去（`os.link`），旧数据集本身不改动、不占额外磁盘                                                                     | 原示教 + 新 episode 全部重新转换（openpi `README_ja.md` 3.3 的方法 2）                                               |
| 速度       | 快，只转新增的                                                                                                                                                           | 慢，几十条示教也要全部重新解码                                                                                         |
| 结果       | 旧 episode 在前、新 episode 在后；只要转换脚本和参数没变，两种做法的内容相同                                                                                             | 同左                                                                                                                   |
| 限制       | 新旧数据的相机 / velocity / effort 必须一致，不一致直接报 `features of ... differ`；旧数据集必须只有一个 chunk，合并后总 episode 数不超过 `chunks_size`（默认 1000） | 没有这些限制                                                                                                           |
| 什么时候用 | 旧的 LeRobot 数据集还在，而且就是用现在这个转换脚本和参数转的                                                                                                            | 旧数据集是用别的版本 / 参数转的（比如上游带 BGR→RGB 的版本）、想剔除或替换某几条旧示教、想改 prompt、旧数据集已经不在 |

两种做法之后都要**重新算 norm stats**（`norm_stats.json` 不会跟着带过去）。不要把整条 HIL episode 直接并进去：约 30% 的帧是暂停（从臂不动），不是想让模型学的东西。openpi `examples/aloha_real/README_ja.md` 3.3 方法 1 里补充 `packaging_move_box` 的命令就是做法 A，用法完全一样。

1. **转换 + 合并进 LeRobot 数据集**（做法 A；`--base-repo-id` 指向的旧数据集不会被改动）：

   HIL 数据是在机器人本机录的，转换在服务器上做，先把切好的目录拷过去：

   ```bash
   rsync -av /home/robot/Dataset_and_Checkpoint/dataset/<task>_hil_corrections/ \
     server:/mnt/ssd2/guningquan/Dataset/<task>_hil_corrections/
   ```

   ```bash
   cd /mnt/ssd2/guningquan/Programs_server/openpi
   uv run examples/aloha_real/convert_aloha_data_to_lerobot_no_bgr2rgb.py \
     --raw-dir /mnt/ssd2/guningquan/Dataset/<task>_hil_corrections \
     --base-repo-id /mnt/ssd2/guningquan/Dataset/packaging_simple_2_lerobot \
     --repo-id /mnt/ssd2/guningquan/Dataset/packaging_simple_2_hil_lerobot \
     --task "packaging the items" --num-workers 6
   ```

   - `--raw-dir` 要求目录里直接是 `episode_N.hdf5`（只匹配 `episode_*.hdf5`，不递归找子文件夹），`hil_split_episodes.py` 切出来的文件名正好是这个格式。
   - **`--repo-id` 如果已经存在，会被整个删掉重建**，所以它必须和 `--base-repo-id` 不同（脚本也会拒绝两者相同）。
   - 转完最后一行会打印 `has N episodes, M frames`，并且脚本已经逐条核对过新 episode 的长度和 hdf5 一致；把 M 和 `hil_compare_lengths.py` 算出的"原数据帧数 + 复制份数 × 新帧数"对一下。
   - **想要复制 k 份**（openpi 没有加权，只能这样抬高占比）：`--raw-dir` 不直接指向切分目录，而是指向一个只放符号链接的目录，每份用不同的编号（下面以 k=4 为例）：

     ```bash
     S=/mnt/ssd2/guningquan/Dataset/<task>_hil_corrections          # 切分目录（已拷到服务器）
     R=/mnt/ssd2/guningquan/Dataset/<task>_hil_corrections_x4       # 转换用目录，里面全是符号链接
     mkdir -p $R; N=$(ls $S/episode_*.hdf5 | wc -l)
     for c in 0 1 2 3; do for i in $(seq 0 $((N-1))); do
       ln -s $S/episode_$i.hdf5 $R/episode_$((c*N+i)).hdf5; done; done
     ```

     然后 `--raw-dir $R`，其余不变。
2. **在 `src/openpi/training/config.py` 里新加一个 `TrainConfig`**：复制现有的 `pi05_packaging_simple_2_plus_full` 那段，只改 `data.repo_id` 指向上一步的新数据集。`weight_loader`/`pytorch_weight_path` 从 0 开始还是接着练，两种都可以：

   - **从 0 开始**：保持指向 pi0.5 官方基础模型（`gs://openpi-assets/checkpoints/pi05_base/params` / `pytorch_weight_path=.../pi05_base_pytorch`）——这个项目里 `pi05_packaging_simple_2_full`/`_plus_full` 这几个配置目前都是这么写的，但这是**这几个 config 自己的选择，不是框架限制**。
   - **接着上一次任务 checkpoint 继续练**：把 `pytorch_weight_path` 改指向旧 checkpoint 的 step 目录，比如 `.../pi05_packaging_simple_2_full/packaging_simple_2_pt/20000`。`scripts/train_pytorch.py` 存 checkpoint 时写的就是一个 `model.safetensors`（第 167 行），加载 `pytorch_weight_path` 时读的也是同名文件（第 445-446 行），格式完全兼容，不用改代码。`config.py` 里也有先例：`pi05_droid_finetune` 的 `weight_loader` 就指向另一个已微调过的 checkpoint（`gs://openpi-assets/checkpoints/pi05_droid/params`），不是最原始的 base model。
     这种方式是**只加载权重**（跟 ACT 的 `--resume_ckpt_path` 同一类），不等于 `--resume`——`--resume` 是接着同一个 `exp_name` 的中断训练，连 optimizer 状态和 step 数一起恢复，针对的是训练中断重跑，不是"接上一个任务的 checkpoint 开始新任务"。
3. **算归一化统计**（合并数据集后必须重新算，不会自动生成）：

   ```bash
   uv run scripts/compute_norm_stats.py --config-name <新 config 名>
   ```

   config 名要写成 `--config-name <名字>`，直接跟在脚本后面写名字会报参数错误。
4. **训练**（多卡用 `torchrun`；`CUDA_VISIBLE_DEVICES`、`PYTORCH_CUDA_ALLOC_CONF`、显存和耗时的参考值见 openpi `examples/aloha_real/README_ja.md` 第 6 节）：

   ```bash
   uv run torchrun --standalone --nnodes=1 --nproc_per_node=<GPU 数> scripts/train_pytorch.py <新 config 名> --exp-name <新实验名>
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
| `--max_tactile_age`                              | 0.2       | 触觉帧本身（按发布器的 `local_read_end` 算的数据年龄）超过这个秒数就暂停策略    |
| `--no_tactile_calibration`                       | 关        | 每条 episode 开始前不给触觉清零                                                 |
| `--tactile_jpeg_quality`                         | 90        | hdf5 格式：发布器以 raw 发来的触觉 rectify/difference 存成 JPEG 的质量          |
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
| 自动 `Paused (stale stream ...)`                      | 某路相机/触觉断流：检查终端 1 的 launch 输出、`rostopic hz /usb_cam_high/image_raw`、`rostopic hz /tactile/left_gripper_outer/raw` |
| `/tactile/calibrate is not available` 后退出           | 触觉 launch 没起或发布器是旧版本：先起 `4arms_tactile_teleop.launch`；暂时不需要清零就加 `--no_tactile_calibration` |
| `needs a terminal (stdin is not a tty)`               | 必须在交互终端里直接运行，不能放后台或重定向 stdin                                                        |
| 从臂动作一顿一顿                                        | 每个 chunk 推理时会停顿一次（与 openpi aloha_real 相同）；推理慢时看服务器 GPU 是否被占                   |
| `follower: command jumped ... limited`                | 策略输出或主臂读数单步跳变被限速，偶发可忽略，频繁出现要查原因                                            |
| `leader Position_P_Gain is [...]` 启动报错            | 上次运行在主臂变软的过程中被强杀，舵机增益（舵机 RAM 寄存器）停在低值；主臂断电重启、再重启 launch 后再跑 |
| 推理超时后按键一段时间没反应                            | 推理是阻塞的，超时要等 `--infer_timeout`，恢复策略时还要重连服务器；这期间从臂保持不动                  |
