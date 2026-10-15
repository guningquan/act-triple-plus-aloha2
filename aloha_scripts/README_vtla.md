# ALOHA → N0-VTLA 数据采集

`record_episodes_aloha2_vtla.py` 输出 `convert_canonical_data.py --robot aloha`
所需的原始 episode 目录。继续使用现有 ROS 遥操作启动流程；键盘控制和主臂力反馈复用
`record_episodes_aloha2.py`，原有 HDF5 采集脚本不受影响。

在配置好 ROS / Interbotix 的采集环境运行：

```bash
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python3 aloha_scripts/record_episodes_aloha2_vtla.py \
  --task_name aloha2_tactile_test \
  --start_idx 0 --end_idx 9 \
  --output_dir /home/robot/Dataset_and_Checkpoint/dataset/aloha2_tactile_test
```

- RIGHT 开始；采集中 RIGHT 提前保存、LEFT 重录、ESC 退出；`b` 重置力反馈基线。
- `--episode_idx 0` 可单独录一条，不与 start/end 同用。未指定索引则自动选择空闲目录。
- `--aloha-action-space eef` 为默认模式，保存20维 EEF CSV，同时保留14维关节 CSV。
  `--aloha-action-space joint_position` 仅保存14维关节状态和指令 CSV，跳过 EEF 正运动学计算。
  也支持下划线别名 `--aloha_action_space`。所选模式写入 `metadata.json`。
- 默认保存路径直接使用任务配置 `dataset_dir`，不添加后缀；已有 episode 默认拒绝覆盖。
- `--overwrite` 显式覆盖同编号目录，导出失败时保留此前完整 episode。
- 需要 `ffmpeg`（含 `libx264`）、`ffprobe`、`modern_robotics` 和原有采集脚本的 ROS 依赖。
- 必须有三路相机和四路触觉 `rectify`。缺失或超过 `--max_stream_age 0.5` 秒未收到消息时，
  报错停止，不保存缺失图像的 episode。启动等待默认 `--stream_timeout 5` 秒。
- 主臂力反馈默认开启，仍需要触觉发布器提供 `force`；`--no_force_feedback` 可以关闭。
- `--save_force` 额外保存四路 `(T,35,20,3)` float32 NPY，需要触觉发布器启用
  `extra_outputs:="force"`。VTLA 当前转换器不读取这些力场文件。

默认图像映射（已按当前安装选择；每个 slot 均有同名命令行参数）：

| VTLA slot（前缀`observation.image.`） | 来源                                     |
| --------------------------------------- | ---------------------------------------- |
| third_view                              | cam_high；可用`--third_camera cam_low` |
| left_wrist_view                         | cam_left_wrist                           |
| right_wrist_view                        | cam_right_wrist                          |
| left_wrist_left_tactile                 | left_gripper_outer                       |
| left_wrist_right_tactile                | left_gripper_inner                       |
| right_wrist_left_tactile                | right_gripper_inner                      |
| right_wrist_right_tactile               | right_gripper_outer                      |

例如交换右夹爪两路时，同时传入：

```bash
--right_wrist_left_tactile right_gripper_outer \
--right_wrist_right_tactile right_gripper_inner
```

默认 `eef` 模式输出结构（`joint_position` 模式没有两个 `eef_pose` 目录）：

```text
aloha2_tactile_test/
└── episode_0/
    ├── metadata.json
    ├── timestamps.csv
    ├── observation.state.eef_pose/data.csv
    ├── actions.eef_pose/data.csv
    ├── observation.state.joint_position/data.csv
    ├── actions.joint_position/data.csv
    ├── observation.image.third_view/video.mp4
    ├── observation.image.left_wrist_view/video.mp4
    ├── observation.image.right_wrist_view/video.mp4
    ├── observation.image.left_wrist_left_tactile/video.mp4
    ├── observation.image.left_wrist_right_tactile/video.mp4
    ├── observation.image.right_wrist_left_tactile/video.mp4
    └── observation.image.right_wrist_right_tactile/video.mp4
```

7路视频均保持各自的**原始尺寸和宽高比**，不做 resize、裁剪或 pad。
例如相机输入640×480就保存640×480，触觉 `rectify` 输入400×700就保存400×700（宽×高）。
编码保持 **30 fps、H.264、CRF 18、yuv420p**；保留尺寸不代表无损压缩。
各路的实际尺寸写入 `metadata.json` 的 `video_sizes`，顺序是 `[width, height]`。
同一路视频在一个 episode 内必须保持尺寸一致；yuv420p要求宽高为正偶数，
不满足时明确报错，不会自动修改尺寸。原尺寸帧会占用更多采集内存和保存空间。
相机显式转 BGR，触觉使用 SDK 的 BGR rectify。所有模态在发送对应动作前取最新值；
导出时按实际 monotonic 采样时间统一零阶保持重采样到30Hz，不通过修改帧率标签改变时长。
同一输出帧的全部字段使用相同源样本索引，CSV行数与视频帧数完全相同。

这仍是异步传感器的“最新值”采样，不是硬件同步。`timestamps.csv` 保存源样本索引、
重采样时间差、各流消息序号、主机接收年龄和可用的 ROS header 时间戳。
触觉没有标准 ROS header 时间戳时记0。序号变化代表收到新消息，不保证SDK内容更新。
`metadata.json` 保存复用消息数、最大接收年龄和具体映射。

EEF 每臂10列：`xyz + R[:,0] + R[:,1] + gripper`，左臂在前。
位置单位为米，参考系分别是两从臂自身的 base frame；工具坐标系使用当前 Interbotix
`robot_des.M` 定义，未额外增加触觉工具 TCP 偏移。M 和 Slist 随数据保存，部署时须使用相同定义。
夹爪观测读取 `/puppet_<side>/joint_states` 中名为 `gripper` 的电机角，
按 constants.py 中各侧开闭角归一化；保留反馈越界值以便检查标定。
夹爪指令裁剪到[0,1]，保存的动作对应发送给从臂的目标。

在装有 `pyarrow` 的 N0-VTLA 环境运行：

```bash
python /mnt/server_ssd2/Programs_server/N0-VTLA/scripts/convert_canonical_data.py \
  /home/robot/Dataset_and_Checkpoint/dataset/aloha2_tactile_test \
  /home/robot/Dataset_and_Checkpoint/dataset/canonical_tactile_task \
  --robot aloha \
  --task "Perform the task"
```

本机 `convert_canonical_data.py` 已同步改为用 `ffprobe` 读取各路视频的实际尺寸，
写入转换后的 `meta/info.json`，视频直接复制，转换时不缩放。
同一数据集内，同一个图像键跨 episode 的尺寸必须一致；旧224×224数据与新原尺寸数据
应分别转换，避免一个 feature 的 shape 无法准确描述两种尺寸。
模型输入尺寸由后续训练/推理的预处理负责。

**当前转换器的动作语义：** 此命令默认使用 `action[t] = state[t]`，即使提供
`actions.eef_pose/data.csv` 也会忽略它。采集脚本保留真实目标动作，但不能改变转换器的选择。
如果希望用下一帧观测作为目标，可加 `--eef-action-from-state-shift-frames 1`
（最后一帧重复末状态）；这依然不是读取动作CSV。
若需读取保存的**关节**指令，可加 `--aloha-action-space joint_position`。
若需读取保存的**EEF**指令，需要另行修改转换器，目前没有相应命令行开关。

关节空间采集与转换示例：

```bash
python3 aloha_scripts/record_episodes_aloha2_vtla.py \
  --task_name aloha2_tactile_test \
  --start_idx 0 --end_idx 9 \
  --output_dir /path/to/raw-aloha-data \
  --aloha-action-space joint_position

python /mnt/server_ssd2/Programs_server/N0-VTLA/scripts/convert_canonical_data.py \
  /path/to/raw-aloha-data \
  /path/to/converted-aloha-data \
  --robot aloha --task "do the task" \
  --aloha-action-space joint_position
```

关节列顺序为左臂6关节＋左夹爪、右臂6关节＋右夹爪；关节角单位为弧度，夹爪为归一化值。
转换后填充至32维，前14维有效。此模式的 action 读取记录的关节指令，
不会使用 EEF 默认模式的 `action=state` 逻辑。转换时必须显式传入对应选项，
转换器不会根据原始 `metadata.json` 自动选择模式。

离线测试（不会初始化ROS或操作机器人）：

```bash
CONVERTER_PYTHON=/path/to/python-with-pyarrow \
VTLA_CONVERTER=/mnt/server_ssd2/Programs_server/N0-VTLA/scripts/convert_canonical_data.py \
python3 -m unittest discover -s tests -p 'test_record_episodes_aloha2_vtla.py' -v
```
