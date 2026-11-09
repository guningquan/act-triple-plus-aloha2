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
触覚launchは起動時にセンサーをゼロ点校正するので、起動中はセンサーに触れないでください。`hil_rollout.py`
も各エピソードの開始前に一度校正します（`--no_tactile_calibration` で無効化）。触覚フレーム自体が
`--max_tactile_age`（既定0.2秒）より古い場合はポリシーを一時停止します。

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

← を押してから介入が始まるまで（黄色の `paused` のまま）は、この行に準備の進行状況も表示されます（遅い方のマスターアーム）: まず `releasing [██████░░░░] 7/12`
（マスターアームが柔らかくなっている最中で、まだ触らなくてよい）、次に `settling [████░░░░░░] steady 0.20/0.50s | move 0.014/0.010 rad (R elbow) | wait 0.6/2.0s`
（マスターアームを静止させる）。バーは全関節が `--settle_tolerance` 以内に収まっている連続時間 `steady` に応じて `--settle_s` まで伸び、いずれかの関節が
許容値を超えて動くと0に戻り、どの関節かが表示されます。満タンになると行全体が赤に変わり、フォロワーが追従を始めます。`wait` が `--settle_timeout_s` に
達した場合も、待たずにそのまま介入が始まります。赤になる前に動かした分はフォロワーには伝わらないので、安定させてから操作してください。
この間の色は黄色系のままです（緑＝ポリシー実行中、赤＝介入中）。詳細は `README_hil.md` を参照。

### キー操作

| キー           | タイミング     | 動作                                                                                                                                                         |
| -------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| →             | 開始待ち       | 本エピソードを開始し、ポリシーがフォロワーを制御                                                                                                             |
| →             | ポリシー実行中 | **一時停止**: フォロワーはその場で保持。マスターアームは2秒以内に自動でフォロワーの位置に移動                                                          |
| ←             | 一時停止中     | **介入開始**: マスターアームを柔らかくする（0.5秒）→ トルクオフ → マスターアームが静止するまで待つ（最大2秒）→ フォロワーがマスターアームに追従開始 |
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

## 4. HILデータで学習を続ける

`hil_split_episodes.py` のデフォルトは `--keep intervention` です。「人間が実際にアームを操作していた」区間だけを残し、
DAggerでモデルに与える修正データとして使います。
学習データに `--keep active` / `autonomous` を使っては**いけません**。`active` は、その時ポリシーが間違えた動作まで
「正解」として与えてしまい、`autonomous` はモデルが既にできることをもう一度学ばせるだけで、どちらも新しい情報がありません。
この2つはデバッグや挙動の確認向きで、学習用ではありません。

> 補足: BerkeleyのDAgger実践（下の「デモ : HILの混合比率の決め方」を参照）では、**介入データだけを使うのが必ずしも最も安定とは限らない**
> ことが分かっています。新しい1ラウンドの中でポリシーが自力で正常に動いた（中断されなかった）区間（`--keep autonomous`。ただし
> **今回新しく録ったもの**であり、古い autonomous データではない）も追加で残し、介入データと同程度の量で混ぜると、介入データのみ・
> エピソード全体のみのどちらよりも良い結果でした。`--keep intervention` だけで学習した結果が不安定な場合に試せる方向で、
> 必ず変更すべきというわけではありません。

切り出したエピソード（`<task>_hil_corrections/episode_K.hdf5` または `episode_K/`）は通常の録画と**完全に同じ形式**です。
以降の学習は2つのルートで進め方が異なります。

### デモ : HILの混合比率の決め方

「標準の比率」はありませんが、実際の事例の数字は憶測よりも頼りになります。

- **Berkeleyチーム**（論文『Scalable Behavior Cloning with Open Data, Training, and Evaluation』、コードは
  [i2rt-robotics/abc](https://github.com/i2rt-robotics/abc)）のDAgger比率: **前ラウンドのデータ80% + 今ラウンドの介入修正10% +
  今ラウンドの残り10%**（同じ新しいエピソード群のうち、ポリシーが自力で正常に動いた、中断されなかった区間）。
  「介入データのみ」と「完全なエピソードのみ」の両極端も試しましたが、**どちらもこの比率より悪い**結果でした。介入データのみだと、
  モデルが「古いデモの場面」と「新しい修正の場面」の間に偽の相関を学習してしまいます。
- **悪い例**（Pebblousのブログ、実際のGR00T学習記録）: 修正エピソード93本（混合データの24%）を**重み付けなしで単純連結**したところ、
  クローズドループの成功率が73%から43%に低下し、offline指標では問題が見えませんでした。**重み付けなしの連結は、修正信号を薄める
  だけでなく、モデルを積極的に悪化させることさえあります**。
- **良い例**（Axis Roboticsのブログ）: おおよそ **2:1（元データ : 新しい修正データ）** で混ぜ、クローズドループの成功率が
  57.1% → 66.3%。

**「重み付けなしの単純連結」とは具体的に何か**: LeRobotの `MultiLeRobotDataset`（openpiがデータセットを統合するときに使うクラス）は、
`__len__()` が総フレーム数を返し、`__getitem__(idx)` はグローバルなフレーム番号でそのまま取り出します。つまり2つのデータの全フレームを
1本の大きなリストに並べ、`DataLoader` がシャッフルして均一に引くので、どのフレームも選ばれる確率は同じです。**確率を調整する仕組みは
コード上に存在せず**、わざと下げているわけではありません。実測（次の節の `hil_compare_lengths.py` を使用）: `packaging_simple_2` の
50本のデモは合計91,616フレーム（1本平均約1,830フレーム、最長2,500）。`packaging_simple_2_hil` の7本のHILエピソードは合計21,322
フレームで、実際に介入していたのは2,489フレームだけ（5本で各約500フレーム、約10秒）、残り2本は介入フレームが0（録画したが実際には
修正しなかったため、切り出しても貢献しない）、さらに約30%（6,487フレーム）は一時停止中でフォロワーが動いていません。介入データは
元データのわずか2.7%で、そのまま統合するとほとんど埋もれてしまいます。

2つの学習ルートはそれぞれ次のように対処します:

- **ACT**: 既存の `sample_weights`（下のACTの節を参照）で、重みによって比率を直接制御できます。ファイルをいじる必要はありません。
- **openpi / pi0.5**: データセット単位の重みフィールドはありません。追加するHILデータを元データの10%〜20%にしたければ（Berkeleyと同じ
  オーダー）、多く録るか、物理的な手段を使います。切り出したHIL修正エピソードを変換前に何部かコピーしておく方法です（下のopenpiの節の
  シンボリックリンクの書き方を参照）。乱暴ですが有効です。統合後は**必ずクローズドループ評価**を行ってください。loss / offline指標
  だけでは不十分です（Pebblousの事例は、offlineでは問題が見えず、実機にデプロイして初めて悪化に気づいたものです）。

「追加データが元データの何％か」（new/base）に換算すると、Berkeleyの10%（旧データ80%に対して）は約12.5%、Axisの2:1は50%です。
new/baseを10%〜20%にするのは、Berkeleyと同じオーダーで、Axisより控えめです。

### まず長さを測り、それから比率を決める: `hil_compare_lengths.py`

手計算せず、2つのデータの長さを直接測れます。追加する介入データが元データの何％か、10% / 20%に届かせるには何部コピーするか、
あとどれだけ録ればよいかが分かります:

```bash
python3 aloha_scripts/hil_compare_lengths.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil
# デフォルトでは新データのうち「人が実際に介入した」部分だけを数えます（--min_frames 25 フレーム以上の区間。
# 基準は hil_split_episodes.py --keep intervention と完全に同じ）
# --ratio 0.1 0.2   希望する 追加/元データ（デフォルトは 10% と 20%）
# --part all        新データをエピソード全体で数える。--part autonomous|active は他の2つの切り方に対応
# --per_episode     各エピソードのフレーム数とセグメント数
```

第1引数は元データで、`episode_*.hdf5` のディレクトリか、変換済みのLeRobotデータセット（`meta/info.json` の `total_frames` を読む）です。
第2引数は `hil_rollout.py` が録画したディレクトリ（分割の前後どちらでも可）です。フレーム数は `/action` で数え、hdf5は50 Hzです。
上の2つのディレクトリでの出力:

```text
base : .../packaging_simple_2  [hdf5]  50 episodes, 91,616 frames = 30.5 min @50 Hz (avg 1,832/episode)
new  : .../packaging_simple_2_hil  [hdf5]  7 episodes, 21,322 frames = 7.1 min
       autonomous 12,281 | paused 6,487 | correcting-skip 65 | intervention 2,489
       counted: --part intervention (segments >= 25 frames) = 5 segment(s) in 5 of 7 episodes = 2,489 frames = 49.8 s

new / base = 2.7%    share of new in base + new = 2.6%
  new/base  10%  ->  k = 4  copies (new/base 10.9%, share of merged 9.8%, 9,956 frames); or record 6,673 more frames (~14 more corrections of 498)
  new/base  20%  ->  k = 7  copies (new/base 19.0%, share of merged 16.0%, 17,423 frames); or record 15,834 more frames (~32 more corrections of 498)
```

読み方: 今は2.7%しかなく、10%にするにはこの5区間を4部コピーするか、あと約14回介入する必要があります。20%には7部のコピーが必要で、
同じ5区間を7回繰り返すことになるため、コピーより多く録るほうがよいです。コピーが5部を超えると、ツールが「繰り返しが多すぎる」と表示します。

### ACT（ローカル、このリポジトリ自身の `imitate_episodes_multi_gpu.py`）

`hil_split_episodes.py` で切り出したhdf5は、そのままACTに与えられ、変換は不要です。形式は `record_episodes_aloha2.py` の元の録画と
完全に同じで、`utils.py` の `find_all_hdf5()` は `*.hdf5` を再帰的に探すだけで、ファイル名やディレクトリ構成は問いません。

修正データと元のデモデータを一緒に学習させるには、`constants.py` に既存の `zip_tie_cotrain` を参考にして cotrain 用のタスク設定を
新規作成することを推奨します:

```python
'packaging_cotrain': {
    'dataset_dir': [DATA_DIR + '/packaging', DATA_DIR + '/packaging_hil_corrections'],  # 検証データの分割に使うのは最初のディレクトリのみ
    'sample_weights': [90, 10],  # 各バッチで各ディレクトリから引く確率（正規化後）。長さの比ではない
    'episode_len': 1600,
    'camera_names': [...],       # 元のタスクと同じ
},
```

**`sample_weights` は「各ディレクトリが選ばれる確率」であり、ディレクトリに何フレームあるかとは無関係です**: `utils.py` の
`BatchSampler` は、まず重みでディレクトリを選び、次にそのディレクトリ内から均一に1フレームを引きます。したがって:

- **`sample_weights` を書かないと、各ディレクトリが50/50になります**。修正データが数千フレームしかなくてもバッチの半分を占め、同じ
  フレームが何度も見られて過学習しやすくなります。
- `[1, 3]` は25% / 75% であり、「データ量が1:3」という意味ではありません。
- 介入データを10%にしたければ `[90, 10]` と書きます（`hil_compare_lengths.py` の出力に対応する p が出ます。追加/元データが10%なら
  p ≈ 9%、20%なら p ≈ 17%）。各修正フレームが学習全体で見られる回数はおおよそ `ステップ数 × batch_size × p ÷ 修正の総フレーム数`
  で、修正フレームが少ないとこの回数がとても大きくなるため、注意が必要です。
- 正規化統計はデフォルトでは `dataset_dir` の全ディレクトリを合わせて再計算されます。古いcheckpointと揃えたい場合は、`zip_tie_cotrain`
  のように `'stats_dir': [DATA_DIR + '/packaging']` を追加して元データだけで計算できます。

あとは通常どおり学習し、`--resume_ckpt_path` で古いcheckpointからウォームスタートします（README.mdの「Policy Training」の節は
まさにこの場面のために書かれています）:

```bash
python imitate_episodes_multi_gpu.py \
    --task_name packaging_cotrain \
    --ckpt_dir <新しい ckpt_dir> \
    --policy_class ACT \
    ...（その他のハイパーパラメータは以前と同じ）\
    --resume_ckpt_path <古い ckpt_dir>/policy_last.ckpt
```

`--resume_ckpt_path` はモデルの重みだけを読み込み、optimizerの状態とステップ数は復元しないので、`--num_steps` には「今回の新しい学習で
何ステップ走らせるか」を指定します。`dataset_stats.pkl`（正規化用の平均・分散）は学習のたびに `--ckpt_dir` に作り直され、evalのときも
新しい ckpt_dir 自身のstatsを読むので、データセットが変わっても新旧のcheckpointで食い違うことはなく、心配は要りません。学習後は通常どおり
`--eval --ckpt_dir <新しい ckpt_dir>` でデプロイでき、追加の変換は不要です。

### LeRobot v3.0（lerobot リポジトリで pi05 / MolmoAct2 / ACT を学習）

lerobot リポジトリの学習には LeRobot v3.0 データセットを使います（openpi が使う v2.1 とは別のバージョン）。以下の 2 つのスクリプトはこのディレクトリにありますが、lerobot を import するので、conda 環境 `aloha` ではなく **lerobot の venv**（Python 3.12）で実行します：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
```

**hdf5（`hil_rollout.py` から切り出した修正区間を含む）→ v3.0**：`--raw-dir` には複数のディレクトリを指定でき、指定した順に 1 つのデータセットにまとめます。同じディレクトリを 2 回書くと 2 回分変換されます（k 回コピーして比率を上げるのと同じ）。アクションは収録時の絶対値のまま保存し、`meta/stats.json` は全データの正確な分位点、`meta/source_episodes.tsv` には各エピソードの元ファイルが記録されます。

```bash
python aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py \
  --raw-dir /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
            /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_v30 \
  --task "packaging the items" --fps 50
```

複数のデータはまとめて変換してください。別々に変換してから `lerobot-edit-dataset merge` で結合すると、LeRobot の結合や統計の再計算はエピソードごとの分位点の包絡しか残さないため（q01 は各エピソードの q01 の最小値）、範囲が広めになります。

**`lerobot-rollout --strategy.type=dagger` で記録した HIL → 既存の v3.0 データセットに追加**：dagger は自分の `rollout_*` データセット（修正 1 回 = 1 エピソード）にだけ書き込み、学習用のデータセットには手を付けません。結合結果は新しいデータセットとして出力し、入力はどれも変更しません：

```bash
python aloha_scripts/merge_hil_into_lerobot_v30.py \
  --base /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 \
  --hil /home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30_hil
```

2 種類のデータセットは特徴量が違うので、スクリプトが先に揃えます：HIL 側の `intervention`（修正モードではすべて True）と `cam_low` を削除し、base 側の velocity/effort を削除し（dagger は記録せず、pi05・MolmoAct2・ACT は `observation.state` しか使わない）、`robot_type` を base のものに統一します。動画は再エンコードせずに連結し、統計は正確な分位点で計算し直し、最後に HIL フレームの割合を表示します。HIL は `--robot.gripper_state_format=aloha2` を付けて記録したものでなければなりません。

**2 種類の HIL を一緒に使う**：まず変換スクリプトでデモと `hil_rollout.py` の修正区間を一緒に v3.0 に変換し、その上に結合スクリプトで dagger のデータセットを加えます。順番は逆にできません（hdf5 は変換スクリプト、LeRobot データセットは結合スクリプトしか通せない）。dagger のデータを逆にこのリポジトリの ACT や openpi に使うこともできません。`hil_compare_lengths.py` が出した k 回分にするには、修正区間のディレクトリを `--raw-dir` に k 回書きます。2 つの方法の比較と全手順は lerobot リポジトリの `examples/aloha_ros/README_JA.md` 10 章を参照してください。

その後 lerobot リポジトリで `lerobot-train --policy.path=<古い pretrained_model> --dataset.root=<新しいデータセット> ...` として既存モデルから続けて学習します（正規化の統計は新しいデータセットのものに自動で置き換わる。`--resume=true` は使わない）。完全なコマンドは lerobot リポジトリの `examples/aloha_ros/README_JA.md` 8 章・10 章を参照してください。

### openpi / pi0.5（サーバー側）

openpiは元のhdf5を直接読まず、学習にはLeRobot形式のデータセットを使います（`serve_policy.py` がデプロイ時に読む `norm_stats.json` も
このデータセットのディレクトリにあります）。以下の手順は、サーバー上で実際のスクリプトと過去のログ
（`packaging_simple_2_plus_pipeline.log`）に照らして確認済みです:

**統合のしかたは2通りあり、1つ目を推奨します**（以下の「新しいエピソード」は切り出した介入区間のことです。コピーの部数は前の節の
`hil_compare_lengths.py` で決めます）:

|          | A. 新しい分だけ変換し、変換済みのLeRobotデータセットに追加する（`--base-repo-id`、推奨）                                                                                                                   | B. 2種類のhdf5をシンボリックリンクで同じディレクトリにまとめ、全体を変換し直す                                                                                                 |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 方法     | 新しいエピソードだけを変換する。旧データセットのparquetはハードリンク（`os.link`）で引き継ぐので、旧データセットは変更されず、追加のディスクも不要                                                         | 元のデモ + 新しいエピソードをすべて変換し直す（openpi `README_ja.md` 3.3 の方法 2）                                                                                          |
| 速度     | 速い。追加分だけ                                                                                                                                                                                             | 遅い。数十本のデモも全部デコードし直す                                                                                                                                         |
| 結果     | 旧エピソードが前、新エピソードが後ろ。変換スクリプトとパラメータが同じなら、どちらの方法でも内容は同じ                                                                                                       | 左に同じ                                                                                                                                                                       |
| 制限     | 新旧のカメラ / velocity / effort が一致している必要があり、一致しないと `features of ... differ` で即エラー。旧データセットは1 chunkのみで、統合後の総エピソード数が `chunks_size`（デフォルト1000）以下 | 上の制限はない                                                                                                                                                                 |
| 使う場面 | 旧LeRobotデータセットが残っていて、しかも現在のこの変換スクリプトとパラメータで変換したもの                                                                                                                  | 旧データセットが別のバージョン／パラメータで変換されている（例: 上流のBGR→RGB付きバージョン）、古いデモの一部を除外・差し替えたい、promptを変えたい、旧データセットがもう無い |

どちらの方法でも、その後は**norm statsの再計算**が必要です（`norm_stats.json` は引き継がれません）。HILのエピソードを丸ごと統合しては
いけません。フレームの約30%は一時停止（フォロワーが動いていない）で、モデルに学ばせたい内容ではないからです。openpi
`examples/aloha_real/README_ja.md` 3.3 の方法 1 で `packaging_move_box` を追加するコマンドが方法 A で、使い方はまったく同じです。

1. **変換 + LeRobotデータセットへの統合**（方法 A。`--base-repo-id` が指す旧データセットは変更されません）:

   HILデータはロボット本機で録画したもので、変換はサーバー上で行うので、先に分割済みのディレクトリをコピーします:

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

   - `--raw-dir` は、ディレクトリ直下に `episode_N.hdf5` が並んでいる必要があります（`episode_*.hdf5` にだけ一致し、サブフォルダは再帰的に探しません）。
     `hil_split_episodes.py` が切り出すファイル名はちょうどこの形式です。
   - **`--repo-id` が既に存在する場合は、ディレクトリごと削除して作り直されます**。そのため `--base-repo-id` とは別のものにする必要があります
     （両者が同じだとスクリプトが拒否します）。
   - 変換が終わると最後の行に `has N episodes, M frames` と表示され、新しいエピソードの長さがhdf5と一致することはスクリプトが1本ずつ
     確認済みです。M を、`hil_compare_lengths.py` で計算した「元データのフレーム数 + コピー部数 × 新しいフレーム数」と突き合わせてください。
   - **k部コピーしたい場合**（openpiには重み付けがないので、比率を上げるにはこの方法しかありません）: `--raw-dir` に分割済みディレクトリを
     直接指定せず、シンボリックリンクだけを置いたディレクトリを指定し、コピーごとに別の番号を付けます（以下はk=4の例）:

     ```bash
     S=/mnt/ssd2/guningquan/Dataset/<task>_hil_corrections          # 分割済みのディレクトリ（サーバーにコピー済み）
     R=/mnt/ssd2/guningquan/Dataset/<task>_hil_corrections_x4       # 変換用ディレクトリ。中身はシンボリックリンクのみ
     mkdir -p $R; N=$(ls $S/episode_*.hdf5 | wc -l)
     for c in 0 1 2 3; do for i in $(seq 0 $((N-1))); do
       ln -s $S/episode_$i.hdf5 $R/episode_$((c*N+i)).hdf5; done; done
     ```

     そのあと `--raw-dir $R` にして、残りは同じです。
2. **`src/openpi/training/config.py` に新しい `TrainConfig` を追加**: 既存の `pi05_packaging_simple_2_plus_full` の部分をコピーし、
   `data.repo_id` だけを前の手順で作った新しいデータセットに変更します。`weight_loader` / `pytorch_weight_path` は、ゼロから始めるか続きから学習するか、どちらも可能です:

   - **ゼロから始める**: pi0.5の公式ベースモデル（`gs://openpi-assets/checkpoints/pi05_base/params` / `pytorch_weight_path=.../pi05_base_pytorch`）を指したままにします。
     このプロジェクトの `pi05_packaging_simple_2_full` / `_plus_full` などの設定は現在すべてこうなっていますが、これは**これらのconfig自身の選択であり、
     フレームワークの制約ではありません**。
   - **前回のタスクのcheckpointから続けて学習する**: `pytorch_weight_path` を古いcheckpointのstepディレクトリ、たとえば
     `.../pi05_packaging_simple_2_full/packaging_simple_2_pt/20000` に変更します。`scripts/train_pytorch.py` がcheckpointを保存するときに書くのは
     `model.safetensors` 1つ（167行目）で、`pytorch_weight_path` を読み込むときに読むのも同名のファイル（445〜446行目）なので、形式は完全に互換で、
     コードの変更は不要です。`config.py` にも前例があり、`pi05_droid_finetune` の `weight_loader` は、最初のベースモデルではなく、既にファインチューニング済みの
     別のcheckpoint（`gs://openpi-assets/checkpoints/pi05_droid/params`）を指しています。
     この方法は**重みだけを読み込む**もの（ACTの `--resume_ckpt_path` と同じ種類）で、`--resume` とは別です。`--resume` は同じ `exp_name` の中断した学習の続きで、
     optimizerの状態とステップ数も一緒に復元します。対象は学習が中断して再実行する場合であり、「前のタスクのcheckpointから新しいタスクを始める」ためのものではありません。
3. **正規化統計を計算**（データセットを統合した後は必ず再計算が必要で、自動では生成されません）:

   ```bash
   uv run scripts/compute_norm_stats.py --config-name <新しい config 名>
   ```

   config名は `--config-name <名前>` の形で指定します。スクリプトの後ろに名前だけを書くと、引数エラーになります。
4. **学習**（複数GPUでは `torchrun`。`CUDA_VISIBLE_DEVICES`、`PYTORCH_CUDA_ALLOC_CONF`、GPUメモリと所要時間の目安は、openpi
   `examples/aloha_real/README_ja.md` の第6節を参照）:

   ```bash
   uv run torchrun --standalone --nnodes=1 --nproc_per_node=<GPU数> scripts/train_pytorch.py <新しい config 名> --exp-name <新しい実験名>
   ```
5. **デプロイ**: 現在と同じで、`serve_policy.py --policy.config=<新しい config 名> --policy.dir=<checkpointディレクトリ>/<新しい実験名>/<step>` です。

この一連の流れ（HIL収集 → 修正区間の切り出し → 学習データへの統合 → 続きの学習／再ファインチューニング → 新しいcheckpointでデプロイ →
収集を続ける）が、標準的なDAggerのループです。

---

## 5. よく使うパラメータ（`hil_rollout.py`。`run_pi_policy.py` のパラメータは第2節と `--help` を参照）

| パラメータ                                         | デフォルト | 説明                                                                                 |
| -------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------ |
| `--action_horizon`                               | 50         | 1回の推論で実行するステップ数                                                        |
| `--infer_timeout`                                | 10         | 1回の推論のタイムアウト（秒）。超えると自動で一時停止                                |
| `--align_s`                                      | 2.0        | 一時停止時にマスターアームをフォロワーに合わせる時間                                 |
| `--release_ramp_s`                               | 0.5        | 介入時にマスターアームを柔らかくする時間。速すぎると感じたら大きくする               |
| `--settle_s` / `--settle_timeout_s`            | 0.5 / 2.0  | マスターアームの静止判定 / 最大待機時間                                              |
| `--absolute_takeover`                            | オフ       | クラッチを使わず、フォロワーがマスターアームの絶対位置に直接追従する                 |
| `--correction_blend_s` / `--correction_skip_s` | 0 / 0.2    | 介入開始時の缓入（直前の指令から）/ interventionとマークしない時間                   |
| `--max_joint_step`                               | 0.1        | フォロワーの1tickあたりの関節最大変化量（rad）。0で無効化                            |
| `--max_stream_age`                               | 0.5        | カメラデータがこの秒数より古くなるとポリシーを一時停止                               |
| `--max_tactile_age`                              | 0.2        | 触覚フレーム自体（パブリッシャーの `local_read_end` から計算したデータ年齢）がこの秒数より古くなるとポリシーを一時停止 |
| `--no_tactile_calibration`                       | オフ       | 各エピソード開始前の触覚のゼロ点校正を行わない                                       |
| `--tactile_jpeg_quality`                         | 90         | hdf5形式: パブリッシャーがrawで送る触覚 rectify/difference をJPEGで保存するときの品質 |
| `--no_force_feedback`                            | オフ       | マスターグリッパーの力フィードバックを無効化                                         |
| `--pedal_device`                                 | なし       | evdevフットペダル（`--pedal_pause KEY_A --pedal_correct KEY_B --pedal_end KEY_C`） |

全パラメータは `python3 aloha_scripts/hil_rollout.py --help` を参照してください。

## 6. よくある問題

| 症状                                                                 | 対処                                                                                                                                                                                                |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Waiting for policy server ...` が延々と再試行される               | サービスが起動していない、またはポートが通っていない: サーバー側でリスニングしているか確認。SSHトンネル + `--host 127.0.0.1` を使う                                                               |
| `ModuleNotFoundError: msgpack` など                                | 環境が有効化されていない: `conda activate aloha`（システムのPythonにはこれらのパッケージは入っていません）                                                                                        |
| `Warming up the policy server` が長く続く                          | openpiが初回推論でコンパイル中。正常な動作。`--warmup_timeout` を超えた場合のみエラーになる                                                                                                       |
| 実行中に自動的に `Paused (inference failed / no reply ...)` になる | サーバー側でエラーまたはタイムアウト: サーバーのターミナルを確認。GPUが混んでいると推論が遅くなるので `--infer_timeout` を大きくするとよい                                                        |
| 自動的に `Paused (stale stream ...)` になる                        | いずれかのカメラ／触覚のストリームが断流している: ターミナル1のlaunch出力、`rostopic hz /usb_cam_high/image_raw`、`rostopic hz /tactile/left_gripper_outer/raw` を確認 |
| `/tactile/calibrate is not available` で終了する                   | 触覚launchが起動していないか、パブリッシャーが古いバージョン: `4arms_tactile_teleop.launch` を先に起動する。ゼロ点校正が当面不要なら `--no_tactile_calibration` を付ける |
| `needs a terminal (stdin is not a tty)`                            | 必ず対話的なターミナルで直接実行する必要があります。バックグラウンド実行やstdinのリダイレクトはできません                                                                                           |
| フォロワーの動作がぎくしゃくする                                     | 各チャンクの推論ごとに一度停止します（openpiのaloha_realと同じ挙動）。推論が遅い場合はサーバーのGPUが混んでいないか確認                                                                             |
| `follower: command jumped ... limited`                             | ポリシーの出力やマスターアームの読み取り値が1ステップで急変したため制限がかかった状態。稀に出る程度なら無視してよいが、頻発する場合は原因を調査すること                                             |
| 起動時に `leader Position_P_Gain is [...]` エラーが出る            | 前回の実行がマスターアームを柔らかくしている最中に強制終了され、サーボのゲイン（サーボのRAMレジスタ）が低い値のまま残っている状態。マスターアームの電源を入れ直し、launchも再起動してから再実行する |
| 推論タイムアウト後、キー入力に少し反応がない                         | 推論はブロッキングであり、タイムアウトまで `--infer_timeout` の時間待つ必要がある。ポリシー再開時はサーバーへの再接続も必要。その間フォロワーは静止したまま                                       |
