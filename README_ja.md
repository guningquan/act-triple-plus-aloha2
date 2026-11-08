# ALOHA / ALOHA2

このリポジトリは、ALOHA / ALOHA2 ロボットシステム向けのテレオペレーション、データセット収集、模倣学習（imitation learning）ワークフローを提供します。**触覚センシングとリーダーグリッパーの力フィードバックはオプション**です。

- **ロボットハードウェアのセットアップ**: お使いの ALOHA ハードウェア用の ROS ドライバと launch ファイルを ROS ワークスペースにインストール・設定してください。アーム名・カメラのトピック・グリッパーのキャリブレーションを実機に合わせてください。
- **オプションの触覚サポート**: 対応する触覚センサーを使うと、リーダーグリッパーの力フィードバックと触覚データの記録が可能になります。

## 📂 リポジトリ構成

| パス                                          | 用途                                                                                                                                             |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `aloha_scripts/constants.py`                | タスク設定テーブル（`TASK_CONFIGS`）、グリッパーのキャリブレーション定数、`START_ARM_POSE`、制御周期（`DT`/`FPS`）                       |
| `aloha_scripts/test_arm_joints.py`          | 読み取り専用: 何かする前に4本のアームすべての `/joint_states` を確認する                                                                       |
| `aloha_scripts/test_gripper_range.py`       | 読み取り専用: 各グリッパーの開閉可動域を測定し、`constants.py` のキャリブレーションに使う                                                      |
| `aloha_scripts/one_side_teleop.py`          | 単腕または双腕のテレオペレーションテスト（記録なし）。CLIフラグでオプションのリーダーグリッパー力フィードバックを調整可能                        |
| `aloha_scripts/check_tactile_feedback.py`   | オプションの触覚サポート: ライブ触覚ビューワー**兼**、片方のグリッパーずつ力フィードバックの感触を調整できるインタラクティブなスライダーUI |
| `aloha_scripts/sleep_plus.py`               | アームを安全な休止姿勢に移動させる（トルクを切る/切らないを選択可）                                                                              |
| `aloha_scripts/record_episodes_aloha2.py`   | 従来のACT形式のHDF5データセット（`episode_N.hdf5`）を記録する                                                                                  |
| `aloha_scripts/replay_episodes.py`          | 記録済みHDF5エピソードの `/action` を実機の puppet アームで再生する（確認プロンプトなし・実際に動く）                                          |
| `aloha_scripts/visualize_episodes_multi.py` | オフライン: qpos/qvel/effort/action の曲線をプロットし、記録済みエピソードのカメラフレームを書き出す                                             |
| `aloha_scripts/check_episode_timing.py`     | オフライン: 記録した各ステップのカメラ・関節・触覚データがほぼ同じ時刻のものか、触覚データが新しかったかを確認する                               |
| `aloha_scripts/tactile_stream.py`           | オプションの触覚サポート: 触覚パブリッシャーのトピックの共有リーダー（データ年齢の判定、ゼロ点校正サービス、runtime config）。単体のコマンドではない |
| `detr/`                                     | モデル定義                                                                                                                                       |
| `imitate_episodes_multi_gpu.py`             | ACT/TactileACT/Diffusion/CNNMLP ポリシーをマルチGPUで学習・評価する。実機評価にはキーボード操作（RIGHT/ESC/LEFT）あり                            |
| `policy.py`                                 | ACT、DiffusionPolicy、CNNMLP、TactileACT 用のアダプタ                                                                                            |
| `aloha_scripts/robot_utils.py`              | 他のスクリプトが共有して使うロボット用ユーティリティ。単体で実行するコマンドではない                                                             |
| `utils.py`                                  | データ読み込みと学習用のヘルパー関数                                                                                                             |

---

## 🏗️ クイックスタートガイド

### 🖥️ ソフトウェアの選択 — OS対応状況

本プロジェクトは以下の構成で動作確認済みです。

- ✅ **Ubuntu 20.04 + ROS 1 Noetic**（完全にテスト・検証済み）

他の構成でも動く可能性はありますが、まだ検証していません。別のセットアップで本プロジェクトを動かせた場合は、ぜひその経験を共有してコントリビュートしてください！🚀

## 🛠️ インストール

```sh
    git clone https://github.com/guningquan/act-triple-plus-aloha2
    cd act-triple-plus-aloha2
    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install -r requirements.txt
    cd detr && pip install -e .
    cd .. && cd robomimic && pip install -e .
    cd ..
```

スクリプトを実行する前に、`aloha_scripts/constants.py` の先頭付近にある `DATA_DIR` を設定してください。

## 1️⃣ ROSを起動し、ハードウェアを確認する

お使いのALOHAハードウェア用のROS launchファイルで、4本のアームとカメラを起動します。
まずROSワークスペースをsourceしてから（`~/interbotix_ws_noetic` が異なる場合は置き換えてください）、
ハードウェア用のlaunchコマンドを実行します。このターミナルは開いたままにしておいてください。

```sh
# ROSターミナル
source /opt/ros/noetic/setup.sh && source ~/interbotix_ws_noetic/devel/setup.sh
# ここにお使いのハードウェアセットアップが提供するアーム/カメラのlaunchコマンドを実行してください。
```

**オプション — 触覚センシングと力フィードバック:** XENSE統合を使用する場合、以下のlaunchコマンドで
アーム・カメラ・触覚パブリッシャーを同時に起動します。

```sh
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:="force"
```

この統合では、`extra_outputs:="force"` によってリーダーグリッパーの力フィードバックや生の力データ記録に
必要な force フィールドが有効になります。`force_norm` を別途追加する必要は**ありません**（これは
`force[..., 2]` を再度publishしているだけです）。触覚ハードウェアを使わない場合は、この触覚用launch
コマンドをスキップし、以下のようにテレオペレーションと記録に `--no_force_feedback` を使ってください。

触覚パブリッシャーはlaunch起動時に全センサーをゼロ点校正するので、**起動中はジェルに触れないでください**。
各センサーのフレームはSDKが返した直後に送信されます（1センサーあたり最大60フレーム/秒）。既定の
rectify + force では既定の `image_codec:=raw` のままにしてください。difference と depth も記録する場合は
`extra_outputs` に追加します。3種類の画像をすべて有効にする場合は `image_codec:=jpeg` の方が追従が良く、
新しいフレームが1センサーあたり毎秒約43から約52〜60に増えます。このリポジトリのスクリプトはどちらの
方式にも対応しています。

```sh
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:=difference,depth,force image_codec:=jpeg
```

launch引数・トピック・測定結果の一覧は、ROSワークスペースの `aloha` パッケージのREADME
（`interbotix_ws_noetic/src/aloha/README.md`）を参照してください。

2番目のターミナルで、何かする前にアームの状態を確認します。

```sh
python aloha_scripts/test_arm_joints.py --compact
python aloha_scripts/test_arm_joints.py --arms puppet_left,puppet_right --loop --rate 2
```

`--compact` は各アームの6関節＋グリッパーを1行で表示します。`--loop` と `--rate N` を併用すると
N Hzで継続更新されます。`Ctrl+C` で停止してください。これらの確認はjoint stateの読み取りのみを
行います。オフラインでの可視化にはロボットの起動は不要です。

オプションのXENSE統合では、`rectify` は常に触覚バンドルに含まれます。追加フィールドは
`extra_outputs` で選択します。空の値は追加フィールドを無効化しますが、Pythonスクリプト側の
フィードバック自体は無効化しません。力フィードバックが不要な場合は明示的に `--no_force_feedback`
を使ってください。

## 2️⃣ グリッパーの可動域をキャリブレーションする（初回、またはハードウェア変更後）

`constants.py` には各グリッパーの開閉関節角の実測値（`MASTER_LEFT_GRIPPER_JOINT_OPEN`、
`PUPPET_RIGHT_GRIPPER_JOINT_CLOSE` など）がハードコードされています。これらはリーダー↔フォロワーの
マッピング、力フィードバックのソフトリミット、初期グリッパー開度を決めるため、実機に合わせる必要が
あります。グリッパーごとに一度実行してください（マスターグリッパーは手で直接動かせますが、puppet
グリッパーは最初に `--release` でトルクフリーにする必要があります）。

```sh
python aloha_scripts/test_gripper_range.py master_left
python aloha_scripts/test_gripper_range.py master_right
python aloha_scripts/test_gripper_range.py puppet_left --release
python aloha_scripts/test_gripper_range.py puppet_right --release
```

各グリッパーを全可動域で動かし、`Ctrl+C` で停止して、表示された `min/closed` と `max/open` の値を
`constants.py` 内の対応する8つの定数にコピーしてください。これは手動編集であり、自動で書き込まれる
ものはありません。`MASTER_LEFT`、`MASTER_RIGHT`、`PUPPET_LEFT`、`PUPPET_RIGHT` それぞれ独立に
`*_CLOSE` を実測の最小値、`*_OPEN` を最大値に設定してください。編集後はPythonスクリプトを再起動して
ください（ROSの再起動は不要です）。フィンガーやリンク、グリッパーモーターを交換したとき、あるいは
リーダー/フォロワーの開度範囲が一致しなくなったときは再キャリブレーションしてください。puppetの
`--release` オプションはグリッパーモーターを再起動し、PWMモードに切り替えてトルクを無効化し、
手で動かせるようにします。

## 3️⃣ テレオペレーションテストとオプションの力フィードバック調整（記録なし）

記録を始める前に、リーダー/フォロワーのミラーリングを試してください。触覚フィードバックなしの
セットアップの場合:

```sh
python aloha_scripts/one_side_teleop.py left --no_force_feedback    # または: right / dual
```

オプションの触覚力フィードバックを使う場合は、触覚パブリッシャーを起動し、`--no_force_feedback` を
省略してください（スクリプトはデフォルトでフィードバックを有効にします）。

スクリプトを実行しているターミナルでキーボードを使用します。

| キー            | 動作                                                                                   |
| --------------- | -------------------------------------------------------------------------------------- |
| `→`          | テレオペレーションを開始／再開する。                                                   |
| `←`          | フォロワーの動作を一時停止し、力フィードバックをゼロにして、リーダーをロックする。     |
| `b`           | 有効時に力フィードバックのベースラインを再ゼロ化する。グリッパーは空のままにしておく。 |
| `ESC` / `q` | 停止してロックし、少し待ってからスリープ姿勢に移動して電源をオフにする。               |

`--start_open_ratio` は初期のグリッパー開度を設定します（デフォルト `1.0`、全開）。
`--disable_gripper_contact_guard` はテレオペレーションのデバッグ用にフォロワーグリッパーの
接触/ストールガードを無効化します。通常使用時はガードを有効にしたままにしてください。

### オプションの触覚フィードバック調整

触覚力フィードバックを使っていて感触がおかしい（強すぎる、弱すぎる、予期せず引っかかる等）場合は、
片方のグリッパーに対してスライダーでリアルタイムに調整できます（アームは事前に安全な休止姿勢に
しておく必要があります）。

```sh
python aloha_scripts/check_tactile_feedback.py left   # または: right
```

この調整ツールは選択したグリッパーのみを制御し、アーム関節の動作やトルクは変更しません。
まずアームを安全な休止姿勢にしてください。例: `python aloha_scripts/sleep_plus.py --sleep`。

10個のスライダーで、力のデッドバンド、フルスケール、カーブ、壁剛性の最小/最大値、最大フィードバック
PWM、立ち上げ時間、クロージング速度制限、クロージングブレーキゲイン、最大ブレーキPWMを制御します。
`1`～`9` または `0` でパラメータを選択し、数値を入力して `Enter` で適用します。`Esc` は
テキスト入力をキャンセル、または編集していない時は終了します。`b` でベースラインを再ゼロ化します。
`p` を押すか終了すると、調整後のコマンドライン引数が表示されます。

以下のフラグはテレオペレーションとHDF5記録スクリプトの両方で共有されます。

| フラグ                                             | 用途                                                   |
| -------------------------------------------------- | ------------------------------------------------------ |
| `--force_deadband`                               | ベースライン周辺の小さな力を無視する。                 |
| `--force_full_scale`                             | フィードバックの正規化に使う力スケールを設定する。     |
| `--wall_stiffness_min`, `--wall_stiffness_max` | 接触壁の剛性範囲を設定する。                           |
| `--max_feedback_pwm`                             | 触覚フィードバックのPWMコマンドを制限する。            |
| `--close_speed_limit`                            | ブレーキをかけるクロージング速度のしきい値を設定する。 |
| `--close_brake_gain`                             | クロージングブレーキの強さを設定する。                 |
| `--force_baseline_samples`                       | 静止時ベースラインの測定に使うサンプル数を設定する。   |
| `--no_force_feedback`                            | リーダーグリッパーの力フィードバックを無効化する。     |

調整後、対応するフラグを記録用コマンドにコピーしてください。チューナーは `--force_curve`、
`--force_ramp_s`、`--close_brake_max_pwm` も表示しますが、これらは `one_side_teleop.py` では
受け付けられるものの、現時点ではHDF5記録のCLIには公開されていません。`one_side_teleop.py` が
共有のデフォルト値を提供します。左右どちらも同じCLIパラメータ値を使い、触覚トピックと測定ベース
ラインのみが別個になります。`--left_*` / `--right_*` のような個別のフィードバック調整フラグは
ありません。

## 4️⃣ タスクを定義する

`aloha_scripts/constants.py` の `TASK_CONFIGS` にエントリを追加（または既存のものを再利用）します。

```python
'packaging': {
    'dataset_dir': DATA_DIR + '/packaging',
    'episode_len': 1600,          # 実際のタスクにかかる時間（制御ステップ数）に合わせて調整
    'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
    # オプション - 生の触覚配列も記録したい場合のみ追加:
    # 'tactile_sensors': ['left_gripper_outer', 'left_gripper_inner',
    #                      'right_gripper_outer', 'right_gripper_inner'],
    # 'tactile_outputs': ['rectify', 'force'],
},
```

`--task_name` は `TASK_CONFIGS` 内のキーと一致させる必要があります。エピソード長やセンサー一覧は
`constants.py` の現在のテーブルを正とみなしてください。既存の例としては、カメラと関節用の
`aloha2_test`、追加の生触覚配列用の `aloha2_tactile_test` / `adapter_insertion` があります。
上記の `packaging` ブロックは一例なので、エピソード長とカメラを自分のタスクに合わせて調整して
ください。

ハードウェアが実際にpublishしているカメラのみを記載してください。一部の古い `zip_tie` 設定には
`gel` カメラが含まれていますが、そのタスクを選択してもセンサー自体は起動しません。力フィードバック
と触覚配列の保存は独立しています。フィードバックを有効にするだけではHDF5エピソードに触覚データは
追加されません。

`zip_tie_cotrain` のような学習専用の設定では、`dataset_dir` のリストに加えて `stats_dir`、
`sample_weights`、`train_ratio` を使うことがあります。HDF5記録スクリプトは単一のデータセット
ディレクトリを要求するため、収集元データセットごとに別の収集用タスクを用意してください。

## 5️⃣ データセットを記録する

```sh
python aloha_scripts/record_episodes_aloha2.py \
    --task_name packaging \
    --start_idx 0 --end_idx 49 \
    --no_force_feedback
```

オプションの触覚力フィードバックを使う場合は `--no_force_feedback` を省略し、触覚パブリッシャーが
force フィールドを提供していることを確認してください。生の触覚配列の記録は別途設定します。
必要な場合のみタスク設定に `tactile_sensors` と `tactile_outputs` を追加し、触覚センサーのない
セットアップでは外しておいてください。

触覚センサーを使う場合（力フィードバック、またはタスク設定に `tactile_sensors` がある場合）:

- 各エピソードの開始時に、フォロワーのグリッパーを開いた状態でパブリッシャー経由（`/tactile/calibrate`）で
  センサーをゼロ点校正します。このときセンサーに触れないでください。`--no_tactile_calibration` で省略できます。
- 触覚フレームが `--max_tactile_age`（既定0.2秒。受信時刻ではなくフレーム自体の年齢で判定）より古くなると、
  記録を止めてそのエピソードを記録し直します。
- 各ステップの触覚フレームはそのステップのカメラ画像・関節状態と同時に取得され、`--tactile_jpeg_quality`
  （既定90）のJPEGで保存されます。パブリッシャーがすでにJPEGで送っているフレームはそのまま保存します。
- 各HDF5エピソードには、ステップごとのタイムスタンプ（`/observations/timestamps`）と、そのゼロ点校正で
  書き出された触覚の runtime config（`/metadata/tactile`）も保存されます。

これにより両アームが開始姿勢に移動し、`→` が押されるまで記録開始を待機します。記録中:
`→` で保存して早めに停止、`←` で破棄してこのエピソードをやり直し、`b` で（有効な場合）力フィード
バックのベースラインを再ゼロ化、`ESC` で全体を中断します。エピソード間でアームは自動的にスリープ
姿勢へ移動し、終了時（または `ESC` 時）には電源をオフにします。

`--start_idx` と `--end_idx` は必須かつ両端を含みます。HDF5エピソードを1つだけ記録する場合は、
両方を同じインデックスに設定してください。現在のバッチループはこの範囲から `--episode_idx` を
割り当てます。エピソードを破棄してもそのインデックスは消費されません。`ESC` を押すと、それまでに
保存したエピソードは保持され、現在の試行のみ破棄されます。記録は右矢印キーで開始し、グリッパーを
握ることでは開始しません。

## 6️⃣ データを確認する

```sh
python aloha_scripts/visualize_episodes_multi.py --dataset_dir <dataset_dir> --episode_idx 0
```

qpos/qvel/effort/action の曲線をプロットし、カメラフレームを書き出します。ロボットは動きません。
複数のインデックスを `--episode_idx 3,5,8` のように、または範囲を `--episode_idx 5-19` のように
指定できます。

記録したエピソードのタイミングを確認するには（データセットのディレクトリ、1つの `episode_N.hdf5`、
VTLAのエピソードディレクトリのいずれにも使えます）:

```sh
python aloha_scripts/check_episode_timing.py <dataset_dir>
```

制御周期、カメラ・関節・触覚のデータ年齢、新しい触覚フレームを得たステップの割合、触覚センサー間の
時間差について p50 / p95 / max を表示し、それぞれ PASS / WARN / FAIL で判定します。どれかが FAIL なら
終了コードは1です。2026年11月より前に記録したエピソードにはタイムスタンプがないため確認できません。

記録された `/action` が実際に実機のアームを正しく駆動することを確認するには、再生することができます
——**これは実際のpuppetアームを確認プロンプトなしで即座に動かします**:

```sh
python aloha_scripts/replay_episodes.py --dataset_dir <dataset_dir> --episode_idx 0
```

再生には `--dry_run` やステップ数制限のオプションはありません。実行前に作業空間と開始姿勢を
確認してください。終了時に両方のpuppetグリッパーを開きますが、アームをスリープ姿勢には移動しません。
必要な場合は後で `sleep_plus.py` を使ってください。

## 7️⃣ ロボットを片付ける

```sh
python aloha_scripts/sleep_plus.py --shut_down          # すべてのロボットを休止姿勢に移動後、トルクをオフにする。
python aloha_scripts/sleep_plus.py --shut_down_puppet   # puppetロボットのみ休止姿勢に移動し、トルクをオフにする。
python aloha_scripts/sleep_plus.py --sleep              # すべてのロボットを休止姿勢に移動するが、トルクはオンのまま。
```

休止動作は、途中姿勢を経てキャリブレーション済みの休止姿勢へ至る2段階の動きを使います。セットアップ
を調整する際はこの2段階のシーケンスを維持してください。すぐに続けて使う場合はセッション間で
`--sleep` を、作業終了時は `--shut_down` を、フォロワーだけ休止・電源オフにしたい場合は
`--shut_down_puppet` を使ってください。HDF5記録スクリプトはエピソード間で既に全アームをスリープへ
移動させます。

## 🧠 ポリシーの学習

```sh
export CUDA_VISIBLE_DEVICES=0,1
python imitate_episodes_multi_gpu.py \
    --task_name packaging \
    --ckpt_dir <checkpoint_dir> \
    --policy_class ACT \
    --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --batch_size 64 \
    --dim_feedforward 3200 --lr 1e-5 --seed 0 \
    --num_steps 100000 --eval_every 2000 \
    --validate_every 2000 --save_every 2000
```

`--policy_class` には `ACT`、`TactileACT`、`ACTNormals`、`Diffusion`、`CNNMLP` を指定できます。
タスクの `TASK_CONFIGS` エントリに `tactile_sensors`/`tactile_outputs` が含まれる場合は
`TactileACT` を使ってください。既存のチェックポイントから学習を継続する場合（例: HIL/修正済み
エピソードを追加した後）は、`--resume_ckpt_path <ckpt_dir>/policy_last.ckpt` を追加してください。

## 📡 ポリシーの評価／デプロイ

```sh
export CUDA_VISIBLE_DEVICES=0
python imitate_episodes_multi_gpu.py \
    --task_name packaging \
    --ckpt_dir <checkpoint_dir> \
    --policy_class ACT \
    --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --batch_size 8 \
    --dim_feedforward 3200 --lr 1e-5 --seed 0 \
    --num_steps 100000 --eval_every 2000 \
    --validate_every 2000 --save_every 2000 \
    --temporal_ensemble --eval --num_rollouts 20
```

実機（解決済みタスク設定で `real_robot=True`）では、各rolloutにキーボード操作が用意されています。
`→` でrolloutを開始、`ESC`/`q` で停止して電源をオフ（`shut_down_puppet_robots()`）、`←` で
このrolloutだけを中断して再度ホーム姿勢に戻り再試行します（`--num_rollouts` には**カウントされ
ません**）。

## トラブルシューティング

| 症状                                                   | 確認事項                                                                                                                                                                                |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 起動時に `ValueError: Unknown user`                  | `constants.py` で `DATA_DIR` のユーザー名／パス選択を設定してください。                                                                                                             |
| リーダー/フォロワーのグリッパー動作範囲が一致しない    | 4つのグリッパーすべてを個別に再キャリブレーションし、8つの開閉定数を更新してください。                                                                                                  |
| オプションの力フィードバックがゼロのまま               | 触覚トピックとそのデータの新しさを確認し、パブリッシャーの `extra_outputs` で `force` を有効にしてください。forceデータが欠落・古い場合は警告が出てフィードバックがゼロになります。 |
| 要求したHDF5触覚フィールドがゼロになっている           | パブリッシャーが設定した `tactile_outputs` のすべてのフィールドを提供しているか確認してください。欠落フィールドは警告され、HDF5記録側でゼロ埋めされます。                             |
| チューナーが生成したフラグが記録スクリプトに拒否される | 上記で示した共有フラグのみをコピーしてください。チューナーの一部のパラメータはHDF5記録のCLIには公開されていません。                                                                     |
| 左右でフィードバックの感触が異なる                     | グリッパーを空にして各ベースラインを再ゼロ化し、センサー/キャリブレーションの差異を確認してください。現時点では左右で同じ調整値を共有しています。                                       |

## 🙏 謝辞

本プロジェクトのコードベースは [ALOHA](https://github.com/tonyzhaozh/aloha) および [ACT](https://github.com/tonyzhaozh/act) をベースに構築されています。
