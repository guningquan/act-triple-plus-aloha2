# ロボットをローカルで起動し、サーバー上のポリシーに接続する（通常実行 / HIL）

ロボット側（本機、ROS Noetic + conda環境 `aloha`、Python 3.8）は観測の収集、行動の実行、介入(テイクオーバー)、
記録のみを担当します。モデルはGPUサーバー上にあり、openpiまたはN0-VTLAリポジトリ自身の
`scripts/serve_policy.py` がロードします。どちらもwebsocket + msgpackで、デフォルトポートは8000です。

```
[サーバー 10.240.77.20]  serve_policy.py  ◀── websocket :8000 ──▶  [本機] run_pi_policy.py / run_vtla_policy.py / hil_rollout.py ── ROS ── アーム + カメラ (+ 触覚)
```

| シナリオ                                                     | 本機のスクリプト       |
| ------------------------------------------------------------ | ---------------------- |
| openpiの通常実行（推論のみの確認、介入なし・データ保存なし） | `run_pi_policy.py`   |
| N0-VTLAの通常実行（同上）                                    | `run_vtla_policy.py` |
| HIL収集（人による介入 + 記録）                               | `hil_rollout.py`     |

HILの設計・データ形式・分割については `README_hil.md` を参照してください。本ファイルは起動方法とキー操作のみ説明します。

---

## 0. 環境

本機のスクリプトはすべてconda環境 `aloha` 上で実行します（msgpack、websockets、h5py_cache、modern_robotics、
evdevは既に含まれているため追加インストール不要）。

```bash
conda activate aloha
```

`constants.py` の `DATA_DIR` はログインユーザー名によってパスが決まります（`ubuntu20` → `/home/robot/Dataset_and_Checkpoint/dataset`）。

## 1. 起動順序（3つのターミナル）

### ターミナル1（本機）: ROS

アームに電源を入れ、4本のアームすべてを休止位置に置いてから:

```bash
# カメラのみ使う場合（openpi pi0.5など）:
roslaunch aloha 4arms_teleop.launch
# 触覚（N0-VTLAでは必須）またはマスターグリッパーの力フィードバックを使う場合:
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:="force"
```

このターミナルは起動したままにしておきます。触覚を起動していない場合、力フィードバックはデータを
受信できず警告が出るだけです（マスターグリッパーは0 PWMのまま）。`--no_force_feedback` を追加しても
構いません。

### ターミナル2（サーバー）: 推論サービスを起動

```bash
ssh server
nvidia-smi                                   # 空いているGPUを探す
```

openpi（pi0.5 packaging、正規化統計は設定内の絶対パスから自動で読み込まれるためコピー不要）:

```bash
cd /mnt/ssd2/guningquan/Programs_server/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_packaging_simple_2_full \
  --policy.dir=/mnt/ssd2/guningquan/Programs_server/openpi_checkpoints/pi05_packaging_simple_2_full/packaging_simple_2_pt/20000
```

N0-VTLA（`--policy.config` はチェックポイント学習時と一致させること: jointなら `vtla_joint_posttrain`、
eefなら `vtla_tactile_posttrain`）:

```bash
cd /mnt/ssd2/guningquan/Programs_server/N0-VTLA
CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port 8000 \
  --policy.config=vtla_joint_posttrain --policy.dir=checkpoints/vtla_joint_posttrain/<exp>/<step>
```

サービスがリスニングを開始したこと（`Creating server ...`）を確認してください。8000番ポートに
接続できない場合はSSHトンネルを使い、本機側のコマンドでは `--host 127.0.0.1` に変更します。

```bash
ssh -N -L 8000:localhost:8000 server         # 本機で別のターミナルを開いて実行し、閉じずに保持する
```

### ターミナル3（本機）: ロボット側

```bash
conda activate aloha
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python3 aloha_scripts/run_pi_policy.py ...   # 通常実行。第2節を参照
python3 aloha_scripts/hil_rollout.py ...     # HIL。第3節を参照
```

**キー操作はこのターミナルでのみ有効です**（フォーカスをこのターミナルに保つ必要があります。方向キーを
模擬するフットペダルも同様です）。

`--pedal_device` でevdevフットペダルを接続する場合、スクリプトはこのデバイスを占有します（ペダルの
入力はターミナルに同時には入力されません）。3つのペダルはそれぞれ →（`--pedal_pause`、デフォルト
KEY_A）、←（`--pedal_correct`、KEY_B）、e（`--pedal_end`、KEY_C）に対応します。開始・保存の
プロンプトでも使用可能です（一時停止ペダル = 開始／保存、介入ペダル = 破棄）。各エピソードの開始時に、
それまでに積み上がったキー入力・ペダルイベントはクリアされます。

`hil_rollout.py` 起動後の流れ（`run_pi_policy.py` はフォロワーのみ動かします。第2節参照）:

1. サーバーに接続（最大30秒待機）。
2. **4本のアームが開始位置に移動**（`opening_ceremony`）。手をアームから離してください。
3. 力フィードバックが有効な場合、マスターグリッパーのゼロ点を標定します。この時マスターグリッパーに触れないこと。
4. **推論のウォームアップ**: 最初の1回だけ実施。openpiは初回推論時にモデルをコンパイルするため数分かかる
   ことがあり、その間ロボットは動きません。最大待機時間は `--warmup_timeout`（900秒）。
5. `Ready. RIGHT = start the policy` と表示されたら → で開始。

すべて終了（またはESC）すると、4本のアームは休止位置に戻りトルクが切れます。Ctrl+Cでも同じ処理が
走ります。

---

## 2. 通常実行（推論のみの確認）

モデルを動かすだけで、介入もデータ保存も行わず、マスターアームは使用しません。openpiは
`run_pi_policy.py`、N0は `run_vtla_policy.py` を使い、両スクリプトのキー操作は同じです。

### openpi: `run_pi_policy.py`

```bash
python3 aloha_scripts/run_pi_policy.py --host 10.240.77.20 --port 8000 --prompt "packaging the items" --num_episodes 10
```

観測・実行方式はopenpi付属の `examples/aloha_real/main.py` と同じです（qpos + 3台のカメラを224に
リサイズ、50ステップごとにブロッキング推論、50 Hz）。さらにフォロワーの可動域制限／tickごとの速度制限
（`--max_joint_step`）、カメラのデータ鮮度チェック（`--max_stream_age`）、初回のウォームアップ推論が
追加されています。

起動後: フォロワーが開始位置に戻る（グリッパーは開く）→ ウォームアップ推論（1回だけ、openpiは初回
コンパイルで数分かかる場合あり）→ → を押すよう表示。

| キー         | 動作                                                                                  |
| ------------ | ------------------------------------------------------------------------------------- |
| →           | 本ラウンドを開始                                                                      |
| ← または r  | 本ラウンドを中止: フォロワーはその場で停止後、開始位置に戻る。→ で次のラウンドを開始 |
| ESC または q | 停止: グリッパーを開き、フォロワーは休止位置に戻ってトルクオフ                        |

- 推論エラー／タイムアウト（`--infer_timeout` 10秒）やカメラの断流時は、← と同じ扱いで本ラウンドが中止されます。
- 1ラウンドの最大長は `--max_episode_steps`（3000 tick = 60秒）。満了すれば1ラウンド完了とみなされ、
  中止したものは `--num_episodes` に数えられません。
- 1回の推論ごとに `--action_horizon`（50）ステップ実行します。

### N0-VTLA: `run_vtla_policy.py`

```bash
python3 aloha_scripts/run_vtla_policy.py --host 10.240.77.20 --port 8000 --action_space joint --prompt "Perform the task."
```

キー操作は上記と同じ（→ で開始、← または r で中止して開始位置に戻る、ESC または q で停止）。加えて
`--debug` モードがあります: Enterを押すたびに1フレーム取得・画像表示（ウィンドウ内で q が次へ、ESCで
スキップ）、推論して行動のshapeを表示しますが、アームは動かしません。入力 q で終了します。

---

## 3. HIL収集

```bash
# openpi、hdf5を /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil/episode_N.hdf5 に保存
python3 aloha_scripts/hil_rollout.py --task_name packaging_simple_2 --policy openpi \
  --host 10.240.77.20 --port 8000 --prompt "packaging the items" --num_episodes 20

# N0-VTLA、VTLA形式の生データディレクトリ <dataset_dir>_hil/episode_N/ に保存
python3 aloha_scripts/hil_rollout.py --task_name aloha2_tactile_test --policy n0_joint \
  --host 10.240.77.20 --prompt "Perform the task." --num_episodes 20
```

`--output_dir` で保存先ディレクトリを変更できます。`--num_episodes` は**保存される**エピソード数です
（破棄したものは数えません）。

各エピソードの長さの上限 `--max_timesteps`（デフォルトはタスクの `episode_len`。packaging_simple_2
なら2500 tick = 50秒）は、**ポリシー実行 + 人による介入**の時間のみをカウントします。一時停止・
位置合わせ・マスターアームを柔らかくする時間はカウントされません。上限に達した時点で介入中だった
場合は `Time limit reached: hand back (LEFT)` と表示され、← で引き渡すまで終了を待ちます。
一時停止を含めた総時間は上限の最大4倍になります。

実行中はプログレスバー1行のみがその場で更新されます: `有効tick / 上限 [経過時間, 現在の段階 | 記録済み フレーム数 | 介入回数 | 直近の推論時間 | 介入時のオフセット]`。一時停止や介入などのイベント情報は
プログレスバーの上に出力されます。

### キー操作

| キー           | タイミング     | 動作                                                                                                                                                         |
| -------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| →             | 開始待ち       | 本エピソードを開始し、ポリシーがフォロワーを制御                                                                                                             |
| →             | ポリシー実行中 | **一時停止**: フォロワーはその場で保持。マスターアームは2秒以内に自動でフォロワーの位置に移動                                                          |
| ←             | 一時停止中     | **介入開始**: マスターアームを柔らかくする（0.3秒）→ トルクオフ → マスターアームが静止するまで待つ（最大1秒）→ フォロワーがマスターアームに追従開始 |
| ←             | 介入中         | **引き渡し**: マスターアームをロックし、一時停止に戻る                                                                                                 |
| →             | 一時停止中     | **ポリシーを再開**                                                                                                                                     |
| e または Enter | いつでも       | 本エピソードを終了 → 保存するか確認                                                                                                                         |
| ESC または q   | いつでも       | 本エピソードを終了 → 保存するか確認 → 終了                                                                                                                 |
| b              | 介入中でない時 | マスターグリッパーの力フィードバックのゼロ点を再標定（標定中は制御ループが一時停止するため、介入中は反応しません）                                           |

エピソード終了後の確認（フォロワーのグリッパーは先に開くため、何かを把持していれば落下します）:

| キー | 動作                                                                     |
| ---- | ------------------------------------------------------------------------ |
| →   | 保存                                                                     |
| ←   | 破棄（同じ番号で再録）                                                   |
| ESC  | 破棄して終了（ESCで本エピソードを終了した場合は、→/←を選んだ後に終了） |

無効なキー入力は実行されません: 実行中に←を押すと先に一時停止するよう表示されます。介入中に→を押すと
先に←で引き渡すよう表示されます。マスターアームがまだ柔らかくなる途中／静止判定中に→を押すと少し待つ
よう表示され、その状態で←を押しても効果はありません。

### 典型的な介入の流れ

1. ポリシー実行中に失敗しそうだと分かったら → **→** を押して一時停止。フォロワーが停止し、マスターアーム
   が自動的にフォロワーの位置に合わせて移動します（約2秒。マスターアームを無理に動かさないこと）。
2. 両方のマスターアームのグリップを握る → **←** を押す。マスターアームは徐々に柔らかくなり（突然は
   落ちません）、静止すると端末に `relative control from here ... largest offset ...` と表示され、
   フォロワーが手の動きに追従し始めます。マスターアームを離した時のわずかな沈み込みはクラッチの
   オフセットで相殺されるため、フォロワーが跳ねることはありません。オフセットが0.25 radを超えると
   警告が出ます。これはマスター/フォロワーの姿勢差が大きいことを意味するので、いったん引き渡して
   一時停止・位置合わせをやり直すのが安全です。
3. 修正が終わったら → **←** で引き渡し（マスターアームがロック）→ **→** でポリシーを再開。これは
   何度でも繰り返せます。
4. タスク完了 → **e** を押し、続けて **→** を押して保存。

各介入の最初の0.2秒（`--correction_skip_s`）は通常どおり記録されますが、intervention（介入）とは
マークされません。デフォルトでは缓入（blend-in）は行いません（`--correction_blend_s 0`）: クラッチ
モードでは介入開始時の最初の指令はフォロワーが保持していた指令そのものであり、元々連続しているため
です。これらのデフォルト値はあなたのlerobotコマンドと一致しています。

### 収集後

```bash
# 介入区間のみ（DAggerデータ）が必要な場合、既存の変換スクリプトでLeRobot形式に変換して学習する
python3 aloha_scripts/hil_split_episodes.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections --keep intervention
```

### 動画に変換して確認する

変換・学習の前に、まず特定の `episode_N.hdf5` をラベル付き動画に変換し、どの部分が自律動作で
どの部分が人による介入かを直感的に確認できます。

```bash
python3 aloha_scripts/hil_episode_to_video.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil/episode_0.hdf5 \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil/episode_0_annotated.mp4
```

各カメラの映像を並べて結合し、枠の色で現在の状態を示します（端末にも同じ統計とセグメントの時間表が
表示されます）:

- 緑 `AUTONOMOUS` — ポリシーが自律で動作中（`hil/phase == 0`）
- 黄 `PAUSED` — ポリシーが一時停止し、判断待ち（`hil/phase == 1`）
- 濃い赤 `CORRECTING` — 既に介入プロセスに入っているが、前 `--correction_skip_s` 以内のためまだ
  interventionとしては数えられていない状態（`hil/phase == 2` かつ `hil/intervention == 0`）
- 明るい赤 `HUMAN` — 人間が実際に介入中で、本当にinterventionとして数えられている部分
  （`hil/intervention == 1`）

`--fps` で出力フレームレートを指定できます（デフォルト50、すなわち `hil_rollout.py` の収集レート）。
通常（HILでない）の `record_episodes_aloha2.py` の録画もこのスクリプトに渡せますが、その場合は
全体が `AUTONOMOUS` としてラベル付けされます。

---

## 4. よく使うパラメータ（`hil_rollout.py`。`run_pi_policy.py` のパラメータは第2節と `--help` を参照）

| パラメータ                                         | デフォルト | 説明                                                                                 |
| -------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------ |
| `--action_horizon`                               | 50         | 1回の推論で実行するステップ数                                                        |
| `--infer_timeout`                                | 10         | 1回の推論のタイムアウト（秒）。超えると自動で一時停止                                |
| `--align_s`                                      | 2.0        | 一時停止時にマスターアームをフォロワーに合わせる時間                                 |
| `--release_ramp_s`                               | 0.3        | 介入時にマスターアームを柔らかくする時間。速すぎると感じたら大きくする               |
| `--settle_s` / `--settle_timeout_s`            | 0.2 / 1.0  | マスターアームの静止判定 / 最大待機時間                                              |
| `--absolute_takeover`                            | オフ       | クラッチを使わず、フォロワーがマスターアームの絶対位置に直接追従する                 |
| `--correction_blend_s` / `--correction_skip_s` | 0 / 0.2    | 介入開始時の缓入（直前の指令から）/ interventionとマークしない時間                   |
| `--max_joint_step`                               | 0.1        | フォロワーの1tickあたりの関節最大変化量（rad）。0で無効化                            |
| `--max_stream_age`                               | 0.5        | カメラデータがこの秒数より古くなるとポリシーを一時停止                               |
| `--no_force_feedback`                            | オフ       | マスターグリッパーの力フィードバックを無効化                                         |
| `--pedal_device`                                 | なし       | evdevフットペダル（`--pedal_pause KEY_A --pedal_correct KEY_B --pedal_end KEY_C`） |

全パラメータは `python3 aloha_scripts/hil_rollout.py --help` を参照してください。

## 5. よくある問題

| 症状                                                                 | 対処                                                                                                                                                                                                |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Waiting for policy server ...` が延々と再試行される               | サービスが起動していない、またはポートが通っていない: サーバー側でリスニングしているか確認。SSHトンネル + `--host 127.0.0.1` を使う                                                               |
| `ModuleNotFoundError: msgpack` など                                | 環境が有効化されていない: `conda activate aloha`（システムのPythonにはこれらのパッケージは入っていません）                                                                                        |
| `Warming up the policy server` が長く続く                          | openpiが初回推論でコンパイル中。正常な動作。`--warmup_timeout` を超えた場合のみエラーになる                                                                                                       |
| 実行中に自動的に `Paused (inference failed / no reply ...)` になる | サーバー側でエラーまたはタイムアウト: サーバーのターミナルを確認。GPUが混んでいると推論が遅くなるので `--infer_timeout` を大きくするとよい                                                        |
| 自動的に `Paused (stale stream ...)` になる                        | いずれかのカメラ／触覚のストリームが断流している: ターミナル1のlaunch出力、`rostopic hz /usb_cam_high/image_raw` を確認                                                                           |
| `needs a terminal (stdin is not a tty)`                            | 必ず対話的なターミナルで直接実行する必要があります。バックグラウンド実行やstdinのリダイレクトはできません                                                                                           |
| フォロワーの動作がぎくしゃくする                                     | 各チャンクの推論ごとに一度停止します（openpiのaloha_realと同じ挙動）。推論が遅い場合はサーバーのGPUが混んでいないか確認                                                                             |
| `follower: command jumped ... limited`                             | ポリシーの出力やマスターアームの読み取り値が1ステップで急変したため制限がかかった状態。稀に出る程度なら無視してよいが、頻発する場合は原因を調査すること                                             |
| 起動時に `leader Position_P_Gain is [...]` エラーが出る            | 前回の実行がマスターアームを柔らかくしている最中に強制終了され、サーボのゲイン（サーボのRAMレジスタ）が低い値のまま残っている状態。マスターアームの電源を入れ直し、launchも再起動してから再実行する |
| 推論タイムアウト後、キー入力に少し反応がない                         | 推論はブロッキングであり、タイムアウトまで `--infer_timeout` の時間待つ必要がある。ポリシー再開時はサーバーへの再接続も必要。その間フォロワーは静止したまま                                       |
