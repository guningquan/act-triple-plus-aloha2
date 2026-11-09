# aloha_scripts 使用手册

## 0. 目录速览

| 脚本 | 一句话说明 |
| --- | --- |
| `test_arm_joints.py` | 只读：检查4条手臂的 `/joint_states` 有没有正常发布 |
| `test_gripper_range.py` | 只读：读取某一个夹爪的实时角度，配合手动扳动做**行程标定** |
| `sleep_plus.py` | 让机械臂归位（sleep）或断电（shutdown） |
| `one_side_teleop.py` | 单臂/双臂遥操作试跑，不录制数据，用来测硬件、调力反馈手感 |
| `check_tactile_feedback.py` | 单侧夹爪遥操作 + 滑杆实时调力反馈参数，找到手感后导出 CLI 参数 |
| `record_episodes_aloha2.py` | 录制经典 ACT 格式的 `episode_N.hdf5` 数据集 |
| `record_episodes_aloha2_vtla.py` | 录制面向 N0-VTLA 的逐 episode 目录（CSV + mp4），供 `convert_canonical_data.py` 转换 |
| `replay_episodes.py` | 读取一条 HDF5 的 `/action`，在真实 puppet 手臂上原样回放 |
| `visualize_episodes_multi.py` | 离线画一条 HDF5 的 qpos/qvel/effort/action 曲线和相机帧，不碰机械臂 |
| `run_vtla_policy.py` | 连接远端 N0-VTLA 策略服务器，在真机上跑推理闭环 |
| `run_pi_policy.py` | 连接远端 openpi 策略服务器，在真机上纯推理验证（不接管、不录制），见 `README_run_robot.md` |
| `hil_rollout.py` | 连接远端 openpi / N0 策略服务器，离合式人工接管 + 录制 HIL 数据，见 `README_run_robot.md` |
| `hil_split_episodes.py` | 把 HIL episode 切成只含接管段的 episode，见 `README_hil.md` |
| `convert_aloha_hdf5_to_lerobot_v30.py` | 把一个或多个 `episode_*.hdf5` 目录转成一个 LeRobot v3.0 数据集（绝对动作，精确分位数统计），用 lerobot 的 venv 运行，见 `README_run_robot.md` 第 4 节 |
| `merge_hil_into_lerobot_v30.py` | 把 `lerobot-rollout` dagger 录的 HIL 数据集并进一个 LeRobot v3.0 数据集（输出新数据集），用 lerobot 的 venv 运行，见 `README_run_robot.md` 第 4 节 |
| `constants.py` | 全局常量：任务配置表 `TASK_CONFIGS`、夹爪开合角度、`START_ARM_POSE` 等 |
| `robot_utils.py` | 被其他脚本调用的底层工具函数库，不单独运行 |

典型顺序：**启动 ROS → 硬件自检 → 标定夹爪行程（只需一次，除非硬件变化）→ 归位 →
遥操作试跑/调手感 → 正式录制 → 录制后校验/回放 → （VTLA 分支）转换数据 → 部署评估**。

## 1. 启动前提：ROS

所有下面的 Python 脚本都要求先起好 ROS（机械臂、相机、触觉都在这一步启动），保持这个终端不关：

```bash
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:="force"
```

- `extra_outputs` 只需要写你要用到的触觉字段。`rectify` 不管写不写都会带上；力反馈只需要
  `force`（`force_norm` 本质是 `force[..., 2]` 的重复发布，不需要单独加，见
  `check_tactile_feedback.py` 的模块说明）。
- 如果不需要力反馈/触觉（比如只想测手臂运动），`extra_outputs:=""` 也能起，但
  `one_side_teleop.py`/`record_episodes_aloha2*.py` 默认会尝试订阅 `force`，订阅不到只是打印
  警告、力反馈保持 0 PWM，不会报错退出。
- `constants.py` 里 `DATA_DIR` 是按 `os.getlogin()` 用户名写死的两条路径；如果你的登录用户名
  不是 `guningquan` 或 `ubuntu20`，第一次跑任何脚本都会直接 `ValueError: Unknown user`，需要先
  在 `constants.py` 顶部加一条你自己的用户名分支。
- **启动 launch 时不要碰触觉传感器**：触觉发布器启动时会先清零（`calibrate_on_start:=true`，默认）。
- 触觉发布方式（2026-11 起）：每帧在 SDK 读出后立刻由该传感器自己的线程发送（`publish_mode:=on_new_frame`，
  默认；最多 `max_rate_hz:=60` 帧/秒，不重复发同一帧）。`/raw` 的 `meta` 里有 `local_read_end`（SDK 交出这帧的
  主机时间）、`timestamp`、`seq`。`image_codec:=jpeg` 可以让 rectify/difference 以 JPEG 发送（默认 `raw`，与以前
  相同）。旧的定时发送可用 `publish_mode:=timer` 恢复。
- raw 还是 jpeg（2026-11-08 在这台电脑上实测）：
  - 默认的 rectify+force 用 `raw`：50 Hz 录制循环 98% 的步拿到新帧，帧龄 p50 9 ms / p95 19 ms / 最大 37 ms。
  - 三种图像（rectify、difference、depth）全开时用 `jpeg`：每个传感器每秒新帧 raw 约 43，jpeg 52–60。
  - 录制脚本两种都能用；存进 HDF5 的都是 JPEG q90，文件一样。jpeg 的好处是录制时内存小（rectify 每帧
    0.84 MB → 约 28 KB），保存时也不用再编码。

  ```bash
  # 三种图像全开，不压缩 / 压缩
  roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:=difference,depth,force
  roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:=difference,depth,force image_codec:=jpeg
  ```

- 发布器每 60 s 打印一行心跳：`<sensor> read .. ms, new ../s, sent ../s`，`new` 是每秒从 SDK 读到的新帧数。
  对比不同设置时加 `stats_log_period_s:=5` 改成每 5 s 一行；不接机械臂只测传感器加 `launch_driver:=false`。
- 相机保持发原始图（`image_raw`）。在同一台电脑上压缩（JPEG q90）实测延时 p50 从 1.9 ms 变成 3.7 ms，4 个相机节点
  CPU 从 28% 变成 96%，只有跨机器传才值得压缩；录制脚本也只读 `image_raw`。
- 全部 launch 参数、话题格式和实测数据见 ROS 工作空间里 `aloha` 包的 README（`interbotix_ws_noetic/src/aloha/README.md`）。

### 1.1 触觉数据在下游怎么用（2026-11 起）

- 所有脚本通过 `tactile_stream.py` 共用触觉订阅：同一进程里每个话题只订阅一次、只解包一次；raw 和 jpeg
  两种发送方式都能直接用。
- **帧龄按数据本身算**（现在 − `local_read_end`），不是按收到消息的时间。录制脚本里触觉帧龄超过
  `--max_tactile_age`（默认 0.2 s）就**停止这一条并重录**；`hil_rollout.py` 会暂停策略；力反馈输出 0 PWM。
- **每条 episode 开始时自动清零**（调用 `/tactile/calibrate`，在力反馈测基线之前，此时夹爪张开、不要碰传感器）；
  `--no_tactile_calibration` 可以关掉。清零后导出的 runtime config 随 episode 一起保存。
  - 找不到 `/tactile/calibrate`（触觉 launch 没起，或发布器是旧版本）时脚本会报错退出。
  - 加了 `--no_tactile_calibration` 时，改用 launch 启动时那次清零导出的 runtime config：最多等 3 s，收不到就打印警告，
    这条 episode 不带 runtime config 保存。
- 力反馈的滤波按时间计算（`--force_filter` 仍是"每 1/60 秒"的系数），手感不受消息频率和控制频率影响。
- 录完可以用 `check_episode_timing.py` 检查时间对齐（见 7.1）。

## 2. 硬件自检

先确认关节数据在正常发布，不涉及任何运动指令：

```bash
python aloha_scripts/test_arm_joints.py --compact
python aloha_scripts/test_arm_joints.py --arms puppet_left,puppet_right --loop --rate 2
```

`--compact` 每条手臂打印一行 6+1 维关节位置；`--loop --rate N` 按 N Hz 连续刷新，Ctrl+C 退出。

## 3. 标定夹爪行程（第一次上机 / 更换夹爪硬件后必做）

`constants.py` 里下面这 8 个常量是每条手臂夹爪的**实测**开合角度，写死在代码里：

```
MASTER_LEFT_GRIPPER_JOINT_OPEN / MASTER_LEFT_GRIPPER_JOINT_CLOSE
MASTER_RIGHT_GRIPPER_JOINT_OPEN / MASTER_RIGHT_GRIPPER_JOINT_CLOSE
PUPPET_LEFT_GRIPPER_JOINT_OPEN / PUPPET_LEFT_GRIPPER_JOINT_CLOSE
PUPPET_RIGHT_GRIPPER_JOINT_OPEN / PUPPET_RIGHT_GRIPPER_JOINT_CLOSE
```

它们支撑了几乎所有跟夹爪相关的逻辑：主从夹爪的角度映射
（`MASTER2PUPPET_LEFT/RIGHT_JOINT_FN`）、遥操作起始张开比例（`--start_open_ratio`）、力反馈的
软限位（`FEEDBACK_LIMIT_MARGIN`）、录制开始的自动阈值等。**这组常量不对，遥操作的开合行程和
起始位置就都会跟手感不一致**，所以必须先标定。左右手是完全独立标定的，不能只测一侧就套用到
另一侧。

标定步骤（对 4 个夹爪各做一次）：

```bash
python aloha_scripts/test_gripper_range.py master_left
python aloha_scripts/test_gripper_range.py master_right
python aloha_scripts/test_gripper_range.py puppet_left --release
python aloha_scripts/test_gripper_range.py puppet_right --release
```

- master 夹爪默认是可以直接用手扳动的，不用加 `--release`。
- puppet 夹爪平时是带电流位置控制、会抵抗外力的，必须加 `--release`（脚本会先重启夹爪电机、
  切到 PWM 模式并断电）才能用手自由扳动。
- 用手把夹爪来回扳到两端行程（完全闭合、完全张开），终端会实时打印当前角度和一个条形图；确认
  已经覆盖到两端后 `Ctrl+C` 退出，脚本会打印这次运动范围里的 `min/closed`、`max/open`、
  `range`。

**这一步只是读数，不会自动写回代码**：把 4 次运行打印出来的 `min/closed` 和 `max/open`，手动
填进 `constants.py` 里对应的 8 个常量（min → `*_CLOSE`，max → `*_OPEN`）。改完保存即可，不需要
重启 ROS。

什么时候需要重新标定：换了夹爪指头/连杆、拆装过夹爪电机，或者遥操作时发现开合行程跟预期对不
上（比如主臂还没松手手就已经撑到头、或者捏到底puppet还没完全闭合）。

## 4. 机械臂归位 / 断电（`sleep_plus.py`）

```bash
python aloha_scripts/sleep_plus.py --sleep            # 全部（master+puppet）归位，torque 保持开启
python aloha_scripts/sleep_plus.py --shut_down         # 全部先归位再断电，今天不再用时执行
python aloha_scripts/sleep_plus.py --shut_down_puppet  # 只让 puppet 归位断电，master 不动
```

三个都是先经过一个此前常用的过渡姿态，再到实测出来的真正静止姿态，两段式移动，避免手臂在最后
一段里自由落体式砸下来（这两段位置和这套顺序是专门测出来的，不要改成一步到位）。

- `--sleep`：两条 episode 之间、或者中途要离开一下但等会儿还要继续用时执行，torque 不断开。
- `--shut_down`：今天用完收工时执行，断电后手臂可以放心离开。
- `--shut_down_puppet`：只想让 puppet 断电（比如接下来要用 `check_tactile_feedback.py`
  单独调 master 的力反馈手感，不需要 puppet 继续跟随）时用，不影响 master。

`record_episodes_aloha2.py` 批量录制时，两条 episode 之间会自动调用 `sleep_all_robots()`
（等价于 `--sleep`），不需要你手动执行。

## 5. 遥操作试跑（`one_side_teleop.py`）

不录制任何数据，纯粹用来测试硬件联动、力反馈手感、双臂协同，正式录制前建议先跑通这一步。

```bash
python aloha_scripts/one_side_teleop.py left    # 只测左臂
python aloha_scripts/one_side_teleop.py right   # 只测右臂
python aloha_scripts/one_side_teleop.py dual    # 左右一起，共用一个键盘
```

键盘控制（在运行脚本的终端里按键，不需要额外开窗口）：

| 按键 | 作用 |
| --- | --- |
| `→`（RIGHT） | 开始，或从暂停中恢复 |
| `←`（LEFT） | 暂停并锁定：从臂停止跟随，力反馈归零，主臂立刻重新锁死 |
| `b` | 重新校准力反馈零点（要求此时夹爪没有夹任何东西） |
| `ESC` 或 `q` | 停止并锁定主臂；1 秒后自动走到 sleep 姿态并断电，不需要再按确认 |

常用参数：

- `--start_open_ratio`（默认 1.0，全开）：遥操作开始前主从夹爪停靠的张开比例。
- `--disable_gripper_contact_guard`：关掉从臂夹爪的堵转/接触保护（调试用，正常不建议关）。
- `--no_force_feedback`：完全关闭主臂力反馈，退回纯位置跟随。
- 力反馈/闭合刹车相关的一整套参数（`--force_deadband`、`--force_full_scale`、
  `--wall_stiffness_min/max`、`--max_feedback_pwm`、`--close_speed_limit`、
  `--close_brake_gain` 等）：直接改这里能立刻感受到手感变化，但每次改完都要重启脚本。想要边捏
  边拖滑杆实时调，用第 6 节的 `check_tactile_feedback.py` 效率更高；调好之后这里和录制脚本用
  的是**完全相同的参数名**，可以直接照抄。

## 6. 力反馈手感调参（`check_tactile_feedback.py`）

只驱动**一侧夹爪**做遥操作（完全不碰机械臂本体的运动/上电状态），把力反馈的每个参数都做成实时
滑杆，一边捏夹爪一边调，不用每次改完重启：

```bash
python aloha_scripts/check_tactile_feedback.py left
python aloha_scripts/check_tactile_feedback.py right
```

运行前提：两条手臂已经处于安全静止状态（比如刚执行过 `sleep_plus.py --sleep`），因为这个工具
从头到尾都不会移动或者上/断电机械臂本体，只接管夹爪。

- 会弹出一个 "haptics controls" 窗口，10 个滑杆分别对应 `deadband`、`full_scale`、`curve`、
  `stiffness_min/max`、`max_feedback_pwm`、`ramp_s`、`close_speed_limit`、`close_brake_gain`、
  `close_brake_max_pwm`，每个滑杆下面都配了一行中文说明这个值是什么、调大调小分别是什么效果。
- 除了拖滑杆，也可以按数字键 `1`-`9`/`0` 选中某个参数，直接打字输入精确数值，回车确认
  （`Esc` 取消这次输入，不在输入状态时 `Esc` 才是退出程序）。
- 按 `b` 随时重新校准零点；按 `p`（或者直接退出程序）会打印一行可以直接复制粘贴的
  `--force_deadband ... --force_full_scale ...` 参数串，跟 `one_side_teleop.py`、
  `record_episodes_aloha2.py`、`record_episodes_aloha2_vtla.py` 用的是同一套参数名，调好直接
  拿去正式录制用。

## 7. 数据录制

### 7.0 任务配置表（`constants.py` 的 `TASK_CONFIGS`）

`--task_name` 只能填这张表里已有的 key（要新增任务就在 `constants.py` 里照着加一项）。
`dataset_dir` 里的 `{DATA_DIR}` 是第 1 节说的、按登录用户名解析出来的根目录。

| task_name | episode_len | camera_names | tactile_sensors | tactile_outputs | dataset_dir |
| --- | --- | --- | --- | --- | --- |
| `zip_tie_data_collection` | 900 | cam_high, cam_low, cam_left_wrist, cam_right_wrist, gel | 无 | 无 | `{DATA_DIR}/zip_tie_random` |
| `zip_tie` | 900 | cam_high, cam_left_wrist, cam_right_wrist, gel | 无 | 无 | `{DATA_DIR}/zip_tie_random` |
| `zip_tie_normals` | 900 | cam_high, cam_left_wrist, cam_right_wrist | 无 | 无 | `{DATA_DIR}/zip_tie_random` |
| `zip_tie_cotrain` | 900 | cam_high, cam_left_wrist, cam_right_wrist, gel | 无 | 无 | 两个目录的列表（见下方说明） |
| `aloha2_test` | 1600 | cam_high, cam_low, cam_left_wrist, cam_right_wrist | 无 | 无 | `{DATA_DIR}/aloha2_test` |
| `aloha2_tactile_test` | 1600 | cam_high, cam_low, cam_left_wrist, cam_right_wrist | left/right_gripper_outer、left/right_gripper_inner（4个） | rectify, force | `{DATA_DIR}/aloha2_tactile_test` |
| `adapter_insertion` | 1600 | cam_high, cam_low, cam_left_wrist, cam_right_wrist | left/right_gripper_outer、left/right_gripper_inner（4个） | rectify, force | `{DATA_DIR}/adapter_insertion` |

- 只有 `aloha2_tactile_test` 和 `adapter_insertion` 配了 `tactile_sensors`/`tactile_outputs`，跑
  这两个 `--task_name` 才会把触觉数组存进 HDF5；其余任务是纯视觉+关节，即使力反馈开着、手上有
  触觉反馈，也不会往 HDF5 里写触觉字段（力反馈和"是否录触觉"是两回事，见第10节）。
- `zip_tie`/`zip_tie_normals` 的 `camera_names` 里 `cam_low`/`gel` 是被注释掉的，说明这两个任务
  目前配的相机集合比 `zip_tie_data_collection` 少，不是遗漏。
- `zip_tie_cotrain` 的 `dataset_dir` 是一个列表（外加 `stats_dir`/`sample_weights`/
  `train_ratio`），这是给多数据集联合训练用的配置项，**不能**直接传给
  `record_episodes_aloha2.py`/`record_episodes_aloha2_vtla.py` 去录制（它们的代码按单个字符串
  路径处理 `dataset_dir`），只在训练脚本里使用。
- `gel` 是否还是这套硬件上实际存在的相机话题，取决于当前 `roslaunch` 有没有带起来对应节点，跟
  这张表本身无关；配了不代表一定能采集到。

### 7.1 经典 ACT HDF5（`record_episodes_aloha2.py`）

```bash
python aloha_scripts/record_episodes_aloha2.py \
  --task_name aloha2_tactile_test \
  --start_idx 0 --end_idx 9
```

- `--task_name` 必须是 `constants.py` 里 `TASK_CONFIGS` 的一个 key，对应了保存路径
  `dataset_dir`、单条 episode 最大步数 `episode_len`、用哪几路相机 `camera_names`，以及可选的
  `tactile_sensors`/`tactile_outputs`（决定是否把触觉原始数组一起存进 HDF5；不是所有任务都配
  了，比如 `aloha2_test` 就是纯视觉任务）。
- `--start_idx`/`--end_idx` 是必填的闭区间，脚本会从 `start_idx` 录到 `end_idx`，中途 LEFT 重
  录不会占用编号，两条 episode 之间自动 `sleep_all_robots()` 归位；全部录完，或者中途 `ESC`
  放弃退出，都会在最后调用 `shut_down_all_robots()` 直接断电（`ESC` 时只保留到上一条已保存的
  episode）。也可以用 `--episode_idx` 单独指定/覆盖某一条，不循环整个区间。
- 脚本先把 master/puppet 移动到 `START_ARM_POSE`，两侧夹爪按第 3 节标定的行程对齐停靠；开始
  录制**是按键盘右方向键，不是捏夹爪触发**（旧版的捏爪触发已经废弃）。
- 录制中按键：`RIGHT` 提前结束并保存，`LEFT` 放弃重录，`ESC` 放弃并整体退出，`b` 重新校准力
  反馈零点。
- 主臂力反馈默认开启，参数就是第 5/6 节调出来的那一套 `--force_*`/`--wall_*`/`--close_*`；
  `--no_force_feedback` 可以整体关掉。
- 触觉（2026-11 起）：每一步的触觉和相机、关节在同一次 `get_observation()` 里取（以前是 `env.step()` 之后才取，
  比相机晚一个控制步）。触觉图像以 JPEG 质量 `--tactile_jpeg_quality`（默认 90，以前是 50）保存；发布器用
  `image_codec:=jpeg` 时直接保存收到的 JPEG，不再二次编码。`--max_tactile_age`、`--no_tactile_calibration`
  见 1.1 节。
- HDF5 里新增 `/observations/timestamps/`（每步的观测时间、相机/关节 header 时间、控制循环时间、每个触觉帧的
  `timestamp`/`read_start`/`read_end`/`frame_age`/`seq`/`new_frame`）和 `/metadata/tactile/`（清零后导出的 runtime
  config、SDK 版本、发布器设置）。布局见 `tactile_hdf5.py`。录完检查：

  ```bash
  python aloha_scripts/check_episode_timing.py <dataset_dir 或 episode_N.hdf5>
  ```

  它统计控制周期、相机/关节延时、触觉帧龄、拿到新触觉帧的比例、4 个传感器之间的时间差，给出 PASS/WARN/FAIL。

### 7.2 面向 N0-VTLA 的录制（`record_episodes_aloha2_vtla.py`）

```bash
python3 aloha_scripts/record_episodes_aloha2_vtla.py \
  --task_name aloha2_tactile_test \
  --start_idx 0 --end_idx 9 \
  --output_dir /home/robot/Dataset_and_Checkpoint/dataset/aloha2_tactile_test
```

跟 7.1 共用同一套遥操作/键盘/力反馈实现，区别主要在**输出格式**：每条 episode 存成一个目录
（`episode_N/`），里面是关节和 EEF 位姿的 CSV，加 7 路视频（3 路相机 + 4 路触觉 `rectify`，各
自保持原始分辨率，30fps H.264），而不是单个 HDF5；这个目录格式是给另一个仓库的
`convert_canonical_data.py --robot aloha` 转换成 N0-VTLA 训练用的 LeRobot 数据集。

- 必须同时有 3 路相机和 4 路触觉 `rectify` 在正常发布。相机/关节缺失或超过 `--max_stream_age`（默认 0.5 s）
  没收到新消息、触觉帧本身超过 `--max_tactile_age`（默认 0.2 s，按数据年龄算）时，停止这一条并重录，不会保存
  残缺的 episode。
- 每条开始时同样自动清零（`--no_tactile_calibration` 关闭），runtime config 存在
  `episode_N/tactile_runtime_config/`。
- `--aloha-action-space eef`（默认）额外保存 20 维 EEF 位姿 CSV；`joint_position` 只存 14 维
  关节 CSV，跳过正运动学计算。
- `--save_force` 可以额外保存 4 路 `(T,35,20,3)` 的原始力场 NPY（需要 roslaunch 时
  `extra_outputs` 里带上 `force`），但目前的转换脚本不读这个。
- 更完整的 slot 映射表、目录结构示例、和转换成 LeRobot 格式的下游命令，见同目录下的
  [`README_vtla.md`](README_vtla.md)，这里不重复。

## 8. 录制后校验 / 回放（经典 HDF5 路径）

**离线可视化**，不碰机械臂，随时可以跑：

```bash
python aloha_scripts/visualize_episodes_multi.py \
  --dataset_dir /path/to/dataset \
  --episode_idx 0
```

画出这条 episode 的 qpos/qvel/effort/action 曲线，并导出相机帧，用来目视检查数据质量。

**真机回放**，会让 puppet 手臂真的按录制的动作序列运动：

```bash
python aloha_scripts/replay_episodes.py \
  --dataset_dir /path/to/dataset \
  --episode_idx 0
```

⚠️ 这个脚本**读取到 `/action` 之后会立刻连接机器人并逐帧下发动作，中间没有任何"按 Enter 确认"
的暂停点，也没有 `--dry_run`/限步数选项**——跟某些其他项目里带二次确认的同名脚本不一样。执行
前务必确认周围没有障碍物、机械臂初始姿态合理。跑完会自动把两个 puppet 夹爪张开，但不会移动到
sleep 姿态，需要的话自己再执行一次 `sleep_plus.py --sleep`。

## 9. VTLA 策略部署（`run_vtla_policy.py`）

用来在真机上跑一个已经训练好、部署在远端的 N0-VTLA 策略（走 websocket + msgpack 协议）：

```bash
# 远端 GPU 机器（N0-VTLA 仓库/环境里）：
python scripts/serve_policy.py \
    --policy.config=vtla_tactile_posttrain \
    --policy.dir=checkpoints/vtla_tactile_posttrain/my_experiment/5

# 本机（接机械臂这台，当前 ROS 环境）：
roslaunch aloha 4arms_tactile_teleop.launch
python3 aloha_scripts/run_vtla_policy.py --host <server_ip> --port 8000 \
    --prompt "Perform the task."
```

- 第一次接一个新服务器/新 checkpoint，先用 `--debug` 跑一遍**纯检查模式**：手臂只会归位一次，
  之后每按一次 `Enter` 采一帧观测、打印各字段 shape、显示 7 张图（`q` 看下一张，`Esc` 跳过），
  发给服务器、打印返回的 action chunk shape，并把整段 chunk 做逆运动学解算打印关节序列
  shape——**全程不会真的下发任何机械臂动作**，适合先确认链路和数据格式对不对，再放开手跑真实
  闭环。
- 真实运行时每隔 `--action_horizon`（默认 50）步重新查询一次策略，`--max_episode_steps`/
  `--num_episodes` 控制单条/总共跑多久多少条。
- 图像/触觉的色彩通道处理跟录制脚本刻意不同（训练数据过了一次视频编解码，推理这里直接用发布器的
  RGB；发布器用 `image_codec:=jpeg` 时先解码再转 RGB），这属于实现细节，正常使用不需要关心，改的话看脚本
  顶部的模块说明。
- 触觉帧本身超过 `--max_tactile_age`（默认 0.2 s）就不用它；每条开始时的触觉基线只用归位之后的新帧。

## 10. 常见坑

- **力反馈参数散落在 3 个脚本里，但都是同一套定义**：`one_side_teleop.py` 是权威默认值来源，
  `record_episodes_aloha2.py`/`record_episodes_aloha2_vtla.py` 只是转发同名参数；改手感请用
  `check_tactile_feedback.py` 现场调，调完把打印出来的那行参数原样传给正式录制命令。
- **左右手当前共用一套力反馈参数**：`build_haptics_config()` 对左右两侧用的是同一份 CLI
  值，唯一按侧区分的只有触觉话题和各自独立测出来的零点（baseline）；如果两只手实际手感明显不
  一致，目前没有 `--left_*`/`--right_*` 单独调的入口。
- **`check_tactile_feedback.py` 完全不碰手臂**：用它之前必须已经把机械臂摆到安全姿态（比如
  `sleep_plus.py --sleep`），它只接管夹爪，不会主动保护/移动手臂本体。
- **`replay_episodes.py` 没有确认提示**：见第 8 节的安全提示，跑之前自己确认好现场安全。
- **`extra_outputs` 决定触觉字段是否可用**：`roslaunch` 时没带 `force`，力反馈会一直是 0 PWM
  并打印警告，不会报错；没带你想录的字段（比如某任务配了 `tactile_outputs` 里有
  `force_norm`）会在开始录制前警告，这个字段不会保存（不再用 0 填充），不影响其他字段正常保存。
