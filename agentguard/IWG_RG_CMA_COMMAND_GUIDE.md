# IWG + RG-CMA 训练与运行命令手册

本文档面向当前 TrackTrack 仓库中的 Safe-Direct IWG + RG-CMA。目标是以后只修改少量变量，就能自行完成训练、验证和 test 提交，不必重新查命令。

本文所有示例均在前台运行，不使用 `nohup`、`setsid` 或后台 `&`。命令末尾的 `tee` 只负责同时把终端输出写入日志，不会把任务放到后台。

快速导航：

- 第 7 节：复用现有 dataset 的训练命令
- 第 8 节：首次准备数据的一键脚本
- 第 9 节：训练日志与 checkpoint
- 第 10 节：`run.py` 参数解释
- 第 11 节：validation/all raw 命令
- 第 12 节：test final+post 命令
- 第 14 节：结果检查与提交打包
- 第 15 节：当前常用 checkpoint 路径
- 第 17 节：最短命令速查

## 1. 先记住这几条

1. 所有命令都从仓库根目录 `/home/shang/workspace/TrackTrack` 开始。
2. 训练固定使用 CUDA；AgentGuard 跟踪推理通常使用 CPU。
3. 训练 seed 固定为 `42`，tracker seed 固定为 `10000`，两者不要混用。
4. 默认训练 batch size 使用 `1024`。
5. validation/all 只跑 raw，绝不加 `--use_post`。
6. test 没有 GT，使用 `--use_post --skip-eval` 生成提交结果。
7. `base` 是 Safe-Direct IWG 原始 gate；`final` 是经过 RG-CMA 修正后的 gate。正式 test 默认使用 `final`。
8. 新训练必须指定新的 checkpoint 目录。代码检测到已有 `iwg_rg_cma_last.pt` 时会拒绝覆盖。
9. `tracker-suffix` 必须使用新的、能辨认的短名字；`iwg-attn` 默认会追加 `_iwg_attn_<base|final>`，不要再把 `final_test` 重复写进 suffix。

训练示例直接从仓库根目录执行，使用 `./.venv/bin/python` 和 `outputs/...` 相对路径；tracker
示例先进入 `3. Tracker`，使用 `../.venv/bin/python` 和 `../outputs/...`。不需要每次重新设置
`ROOT`、`PY` 或 `PYTHONPATH`。需要把命令拆成脚本、追加 resource log 时，再按脚本自身的变量写法执行。

## 2. 一次实验的标准流程

```text
已有/新建六事件 dataset
        |
        v
train_iwg_attn（CUDA）
        |
        +--> checkpoints/iwg_rg_cma_epochXXX.pt
        |
        v
run.py validation/all（CPU、raw、有 GT，直接打印五项指标）
        |
        v
选定 epoch
        |
        v
run.py test（CPU、final、post、无 TrackEval）
        |
        v
检查 txt 并打包 ZIP
```

有两种开始训练的方法：

- 首次准备某个数据集：运行仓库已有的一键脚本。脚本会准备 detection cache、event cache、标签和六事件 dataset，再训练。
- dataset 已经存在：直接运行 `agentguard.cli train_iwg_attn`。这是重复实验或比较分片策略时的推荐方法，不会重复生成数据。

## 3. 术语说明

### 3.1 base gate 与 final gate

| 参数 | 实际含义 | 是否使用 RG-CMA |
| --- | --- | --- |
| `--iwg-attn-output base` | Safe-Direct IWG 输出的原始 motion/appearance gate | 否 |
| `--iwg-attn-output final` | `base gate + RG-CMA 有界 correction` | 是 |

同一个 combined checkpoint 同时包含 base IWG 和 RG-CMA。因此比较 base/final 时必须使用同一个 checkpoint，只改变 `--iwg-attn-output`。

### 3.2 raw 与 post

| 类型 | 命令差异 | 用途 |
| --- | --- | --- |
| raw | 不写 `--use_post` | validation/all 选 epoch、做公平消融 |
| post | 加 `--use_post` | test 提交 |

当前 `run.py` 的后处理如下：

| 数据集 | `--use_post` 的实际处理 |
| --- | --- |
| MOT17 / MOT20 | GBI：最长 30 帧缺口插值，再做 gradient boosting smoothing，`tau=12` |
| SportsMOT | 删除少于 5 个检测帧的短轨迹，并线性插值不超过 20 帧的内部缺口 |

注意：SportsMOT 当前默认 post 只有短轨迹过滤和线性插值。

### 3.3 validation、all 与 test

| 数据集 | 有标签评测模式 | test 模式 |
| --- | --- | --- |
| MOT17 | `--mode all`，评测 7 个 FRCNN train 序列 | `--mode test` |
| MOT20 | `--mode all`，评测 4 个 train 序列 | `--mode test` |
| SportsMOT | `--mode val` | `--mode test` |

`MOT17 --mode all` 和 `MOT20 --mode all` 在本仓库里指对应 train 序列的全量有标签评测，不是官方 test。

## 4. 当前可复用的数据位置

以下目录已经生成过，可以直接作为 `--dataset-dir`：

| 训练数据 | dataset 目录 |
| --- | --- |
| MOT17 FRCNN train/all | `outputs/agentguard/experiments/iwg_rg_cma_v1_trainall_seed42_bs1024_native_log/dataset` |
| MOT20 train/all，NSA 事件 | `outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024/dataset` |
| SportsMOT train | `outputs/agentguard/experiments/iwg_rg_cma_v1_sportsmot_train_data/dataset` |
| SportsMOT train+val | `outputs/agentguard/experiments/iwg_rg_cma_v1_sportsmot_trainval_seed42_bs1024/dataset` |

统一缓存位置：

```text
outputs/agentguard/detection_cache
outputs/agentguard/event_cache_v3_iwg_v2
```

所有仍使用的 IWG 标签统一存放在：

```text
outputs/agentguard/labels/iwg_rg_cma/MOT17/nsa_candidate_a_json
outputs/agentguard/labels/iwg_rg_cma/MOT20/nsa_v3_compact
outputs/agentguard/labels/iwg_rg_cma/SportsMOT/nsa_train_v3_compact
outputs/agentguard/labels/iwg_rg_cma/SportsMOT/nsa_trainval_v3_compact
```

SportsMOT 的 train 与 train+val 是两套不同范围，不能互相覆盖。当前标签构建只支持 NSA 事件和 candidate A。

开始训练前可以这样检查 dataset 是否存在：

```bash
test -f outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024/dataset/metadata.json
./.venv/bin/python -c "import json; x=json.load(open('outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024/dataset/metadata.json')); print(json.dumps(x, indent=2, sort_keys=True))"
```

## 5. 训练参数

### 5.1 从头训练固定参数

```text
--device cuda
--epochs 100 或 200
--batch-size 1024（也允许 2048）
--num-workers 4
--lr 0.0001
--weight-decay 0.0001
--warmup-epochs 1
--grad-clip 1.0
--seed 42
AMP=false（CLI 内部固定，不需要传参数）
```

从头训练只允许 100 或 200 个名义 epoch。100e 保存 epoch 25/50/75/100；200e 保存 epoch 50/100/150/200。每轮还会更新 `iwg_rg_cma_last.pt`。

若 batch size 设为 `2048` 并发生 CUDA OOM，训练器唯一的自动回退是 `1536`。为了与现有旧 IWG 实验公平比较，本文示例统一使用 `1024`。

### 5.2 warm-start 固定参数

warm-start 当前只允许以下配置；`epochs` 可以是 25 或 50，且必须使用同样长度的单一全量 phase：

```text
batch_size=1024
lr=0.00001
memory_shards=1
epochs_per_shard=epochs
shard_cycles=1
```

warm-start 只严格加载模型权重，不恢复源实验的 optimizer、scheduler、epoch 或 normalization。新数据集使用自己的 normalization，新的 optimizer 和 scheduler 从头开始。25e 保存 epoch 5/10/25；50e 额外保存 epoch50。

## 6. 分片训练怎么理解

训练参数必须满足：

```text
memory_shards * epochs_per_shard * shard_cycles = nominal epochs
effective full-data epochs = epochs_per_shard * shard_cycles
```

`memory_shards=5` 表示把每个序列的时间轴分别切成 5 个连续的 20% 分片，而不是随机抽取整个数据集的 20%。同一个 shard phase 里包含所有序列对应的那一段。

| 名称 | 参数 | 实际顺序 | 100e 的等效全量 epoch |
| --- | --- | --- | --- |
| 旧方式 | `5 * 10 * 2` | 第 1 个 20% 连续训练 10 轮，再换第 2 个；全部走完后重复一次 | 20 |
| 新方式 | `5 * 1 * 20` | 每个 20% 只训练 1 轮，快速走完整个数据集，再循环 20 次 | 20 |

SportsMOT train+val 使用 10 个 10% 分片：

| 名称 | 参数 | 200e 的等效全量 epoch |
| --- | --- | --- |
| 旧方式 | `10 * 10 * 2 = 200` | 20 |
| 新方式 | `10 * 1 * 20 = 200` | 20 |

因此日志中的 100/200 是“名义分片 epoch”，不是看过 100/200 次全量数据。比较训练量时应看日志中的 `effective_full_epochs_completed`。

旧方式会让一个连续时间段连续接受多个相近学习率的更新；新方式让不同时间段更频繁地交替出现。二者总样本访问量可以相同，但优化轨迹不同，所以效果不保证相同。

## 7. 通用训练模板

以下模板只训练，不重新生成 cache、标签或 dataset：

```bash
set -euo pipefail
mkdir -p outputs/agentguard/experiments/替换为新的实验名/checkpoints \
         outputs/agentguard/experiments/替换为新的实验名/logs

./.venv/bin/python -u -m agentguard.cli train_iwg_attn \
  --dataset-dir outputs/agentguard/experiments/替换为数据目录/dataset \
  --checkpoint-dir outputs/agentguard/experiments/替换为新的实验名/checkpoints \
  --device cuda \
  --epochs 100 \
  --batch-size 1024 \
  --num-workers 4 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --warmup-epochs 1 \
  --grad-clip 1.0 \
  --seed 42 \
  --memory-shards 5 \
  --epochs-per-shard 1 \
  --shard-cycles 20 \
  2>&1 | tee outputs/agentguard/experiments/替换为新的实验名/logs/train.log
```

每次只需要检查数据目录、实验名、总 epoch，以及三个分片参数。

### 7.1 MOT17 从头训练 100e

MOT17 数据量可直接使用完整 dataset，不需要低内存分片：

```bash
set -euo pipefail
mkdir -p outputs/agentguard/experiments/iwg_rg_cma_mot17_new_seed42_bs1024_100e/checkpoints \
         outputs/agentguard/experiments/iwg_rg_cma_mot17_new_seed42_bs1024_100e/logs

./.venv/bin/python -u -m agentguard.cli train_iwg_attn \
  --dataset-dir outputs/agentguard/experiments/iwg_rg_cma_v1_trainall_seed42_bs1024_native_log/dataset \
  --checkpoint-dir outputs/agentguard/experiments/iwg_rg_cma_mot17_new_seed42_bs1024_100e/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 4 \
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 \
  --memory-shards 1 --epochs-per-shard 100 --shard-cycles 1 \
  2>&1 | tee outputs/agentguard/experiments/iwg_rg_cma_mot17_new_seed42_bs1024_100e/logs/train.log
```

这里 100 个名义 epoch 就是 100 个等效全量 epoch。

### 7.2 MOT20 新分片方式训练 100e

仓库中已经有一套完整的新方式权重（`5×1×20`），见第 15 节；下面命令只在需要重新训练一套新实验时使用，不能复用已有 checkpoint 目录。

```bash
set -euo pipefail
mkdir -p outputs/agentguard/experiments/iwg_rg_cma_mot20_new_interleaved_seed42_bs1024_100e/checkpoints \
         outputs/agentguard/experiments/iwg_rg_cma_mot20_new_interleaved_seed42_bs1024_100e/logs

./.venv/bin/python -u -m agentguard.cli train_iwg_attn \
  --dataset-dir outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024/dataset \
  --checkpoint-dir outputs/agentguard/experiments/iwg_rg_cma_mot20_new_interleaved_seed42_bs1024_100e/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 4 \
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 \
  --memory-shards 5 --epochs-per-shard 1 --shard-cycles 20 \
  2>&1 | tee outputs/agentguard/experiments/iwg_rg_cma_mot20_new_interleaved_seed42_bs1024_100e/logs/train.log
```

若要复现旧方式，只把最后一行改成：

```bash
  --memory-shards 5 --epochs-per-shard 10 --shard-cycles 2 \
```

### 7.3 SportsMOT train 新分片方式训练 100e

```bash
set -euo pipefail
mkdir -p outputs/agentguard/experiments/iwg_rg_cma_sportsmot_train_new_interleaved_seed42_bs1024_100e/checkpoints \
         outputs/agentguard/experiments/iwg_rg_cma_sportsmot_train_new_interleaved_seed42_bs1024_100e/logs

./.venv/bin/python -u -m agentguard.cli train_iwg_attn \
  --dataset-dir outputs/agentguard/experiments/iwg_rg_cma_v1_sportsmot_train_data/dataset \
  --checkpoint-dir outputs/agentguard/experiments/iwg_rg_cma_sportsmot_train_new_interleaved_seed42_bs1024_100e/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 4 \
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 \
  --memory-shards 5 --epochs-per-shard 1 --shard-cycles 20 \
  2>&1 | tee outputs/agentguard/experiments/iwg_rg_cma_sportsmot_train_new_interleaved_seed42_bs1024_100e/logs/train.log
```

### 7.4 SportsMOT train+val 新分片方式训练 200e

train+val 已包含 validation 数据，适合最终刷 test，但不能再把 val 指标当成未见数据的泛化指标。

```bash
set -euo pipefail
mkdir -p outputs/agentguard/experiments/iwg_rg_cma_sportsmot_trainval_new_interleaved_seed42_bs1024_200e/checkpoints \
         outputs/agentguard/experiments/iwg_rg_cma_sportsmot_trainval_new_interleaved_seed42_bs1024_200e/logs

./.venv/bin/python -u -m agentguard.cli train_iwg_attn \
  --dataset-dir outputs/agentguard/experiments/iwg_rg_cma_v1_sportsmot_trainval_seed42_bs1024/dataset \
  --checkpoint-dir outputs/agentguard/experiments/iwg_rg_cma_sportsmot_trainval_new_interleaved_seed42_bs1024_200e/checkpoints \
  --device cuda --epochs 200 --batch-size 1024 --num-workers 4 \
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 \
  --memory-shards 10 --epochs-per-shard 1 --shard-cycles 20 \
  2>&1 | tee outputs/agentguard/experiments/iwg_rg_cma_sportsmot_trainval_new_interleaved_seed42_bs1024_200e/logs/train.log
```

### 7.5 MOT20 权重 warm-start 到 MOT17 训练 25e

```bash
set -euo pipefail
mkdir -p outputs/agentguard/experiments/iwg_rg_cma_mot20e050_to_mot17_new_finetune25_seed42_bs1024/checkpoints \
         outputs/agentguard/experiments/iwg_rg_cma_mot20e050_to_mot17_new_finetune25_seed42_bs1024/logs

./.venv/bin/python -u -m agentguard.cli train_iwg_attn \
  --dataset-dir outputs/agentguard/experiments/iwg_rg_cma_v1_trainall_seed42_bs1024_native_log/dataset \
  --checkpoint-dir outputs/agentguard/experiments/iwg_rg_cma_mot20e050_to_mot17_new_finetune25_seed42_bs1024/checkpoints \
  --device cuda --epochs 25 --batch-size 1024 --num-workers 4 \
  --lr 0.00001 --weight-decay 0.0001 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 \
  --memory-shards 1 --epochs-per-shard 25 --shard-cycles 1 \
  --init-checkpoint outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024_shard20x2/checkpoints/iwg_rg_cma_epoch050.pt \
  2>&1 | tee outputs/agentguard/experiments/iwg_rg_cma_mot20e050_to_mot17_new_finetune25_seed42_bs1024/logs/train.log
```

## 8. 首次准备数据的一键脚本

这些脚本适合 cache、标签或 dataset 尚未准备时使用：

| 用途 | 命令 |
| --- | --- |
| MOT17 100e 完整历史 pipeline | `bash scripts/agentguard/run_iwg_rg_cma_100e.sh` |
| MOT20 100e 旧分片方式 | `bash scripts/agentguard/run_iwg_rg_cma_mot20_100e.sh` |
| SportsMOT train+val 可配置训练 | `bash scripts/agentguard/run_iwg_rg_cma_sportsmot_trainval_100e.sh` |
| SportsMOT train+val 200e 旧分片方式 | `bash scripts/agentguard/run_iwg_rg_cma_sportsmot_trainval_200e.sh` |
| MOT20 e050 到 MOT17 warm-start 25e | `bash scripts/agentguard/run_iwg_rg_cma_mot20_to_mot17_finetune25.sh` |
| MOT17 50e + SportsMOT/MOT20 分片调参（严格串行） | `bash scripts/agentguard/run_iwg_rg_cma_overnight_tuning.sh` |

MOT17 历史脚本默认 batch size 是 `2048`。若要按本文的 `1024` 规范首次运行，必须显式覆盖，并使用新的实验名：

```bash
set -euo pipefail
cd /home/shang/workspace/TrackTrack

RUN_NAME=iwg_rg_cma_mot17_new_seed42_bs1024_100e \
TRAIN_BATCH_SIZE=1024 \
bash scripts/agentguard/run_iwg_rg_cma_100e.sh
```

例如首次准备 SportsMOT train+val，并按新方式训练 200e：

```bash
set -euo pipefail
cd /home/shang/workspace/TrackTrack

RUN_NAME=iwg_rg_cma_sportsmot_trainval_new_run_seed42_bs1024_200e \
EPOCHS=200 \
MEMORY_SHARDS=10 \
EPOCHS_PER_SHARD=1 \
SHARD_CYCLES=20 \
bash scripts/agentguard/run_iwg_rg_cma_sportsmot_trainval_100e.sh
```

注意：部分早期一键脚本还会自动运行历史的 raw+post 验证矩阵。当前规则是 validation 只看 raw；要严格遵守当前实验规范，建议在数据已准备好后使用第 7 节的直接训练命令，再使用第 10 节的 raw 命令评测。

## 9. 训练产物与日志怎么看

下面把 `<实验名>` 替换成实际实验目录名；命令从仓库根目录执行。

主要文件：

| 文件 | 内容 |
| --- | --- |
| `outputs/agentguard/experiments/<实验名>/checkpoints/training.log` | 面向人阅读的逐 epoch 日志，包含 gate acc、motion MAE、appearance MAE、loss、用时等 |
| `outputs/agentguard/experiments/<实验名>/checkpoints/metrics.jsonl` | 每个 epoch 一条结构化 JSON，适合脚本汇总 |
| `outputs/agentguard/experiments/<实验名>/checkpoints/training_summary.json` | 总训练量、总时间、最终 checkpoint、等效全量 epoch |
| `outputs/agentguard/experiments/<实验名>/checkpoints/iwg_rg_cma_last.pt` | 最后一轮权重 |
| `outputs/agentguard/experiments/<实验名>/checkpoints/iwg_rg_cma_epochXXX.pt` | 固定里程碑权重 |
| `outputs/agentguard/experiments/<实验名>/logs/train.log` | CLI 的完整 stdout/stderr |

查看训练日志：

```bash
tail -n 80 outputs/agentguard/experiments/<实验名>/checkpoints/training.log
```

训练正在运行时持续查看：

```bash
tail -f outputs/agentguard/experiments/<实验名>/checkpoints/training.log
```

检查 checkpoint 能否严格加载，并在 dataset 上跑少量 batch：

```bash
./.venv/bin/python -u -m agentguard.cli validate_iwg_attn_checkpoint \
  --checkpoint outputs/agentguard/experiments/<实验名>/checkpoints/iwg_rg_cma_last.pt \
  --dataset-dir outputs/agentguard/experiments/<数据集实验名>/dataset \
  --device cpu \
  --max-batches 8 \
  --output outputs/agentguard/experiments/<实验名>/checkpoint_validation.json
```

这只是模型/数据契约和离线 gate 指标检查，不等于 tracker 的 HOTA/MOTA/IDF1 验证。

## 10. run.py 通用规则

`run.py` 依赖相对路径，必须先进入 tracker 目录：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
```

最短运行命令只需要下面这些参数：

```text
--agentguard-mode iwg-attn
--agentguard-checkpoint ../outputs/agentguard/experiments/<实验名>/checkpoints/iwg_rg_cma_epochXXX.pt
--iwg-attn-output final
--tracker-suffix "唯一名称"
```

`--agentguard-checkpoint` 可以使用相对路径；它相对于当前 shell 工作目录解析。`--seed 10000`、`--kf-type nsa`、`--iwg-attn-output final`、`--agentguard-device cpu` 和 `--detection-cache-root ../outputs/agentguard/detection_cache` 都是当前默认值。resource log、`tee` 和 `--profile-every` 只是可选审计项。

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--dataset` | `MOT17`、`MOT20` 或 `SportsMOT` |
| `--mode` | 选择 train/all、val 或 test 对应的数据和检测缓存 |
| `--sequences` | 可选；只跑指定序列。MOT17/MOT20 的 all 示例显式列出，便于审计 |
| `--agentguard-mode iwg-attn` | 启用 IWG + RG-CMA combined checkpoint |
| `--agentguard-checkpoint` | 单个 `.pt` 权重文件路径，可以是相对于 `3. Tracker` 的 `../outputs/...`；当前不接受 checkpoint 目录 |
| `--iwg-attn-output` | `base` 或 `final` |
| `--agentguard-device cpu` | AgentGuard 小模型在 tracker 内逐事件推理；CPU 通常比频繁 CPU/GPU 同步更合适 |
| `--tracker-suffix` | 决定输出目录名，务必唯一 |
| `--legacy-output-naming` | 可选；仅旧 pipeline 使用，恢复历史的 `_agentguard_iwg_attn_...` 目录拼法 |
| `--print-per-sequence-metrics` | 有 GT 时除总指标外打印每序列指标 |
| `--use_post` | 生成并使用 post 结果；只给 test 命令使用 |
| `--skip-eval` | 跳过 TrackEval；test 没有 GT 时使用 |
| `--resource-log` | 可选的内存和 IO 采样 JSONL |

## 11. validation/all：只跑 raw

以下命令都没有 `--use_post`，也没有 `--skip-eval`。运行结束会直接打印 HOTA、MOTA、IDF1、DetA、AssA。

### 11.1 MOT17 all raw

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset MOT17 --mode all \
  --sequences \
    MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN \
    MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_trainall_seed42_bs1024_native_log/checkpoints/iwg_rg_cma_epoch100.pt" \
  --iwg-attn-output final \
  --tracker-suffix mot17_e100_final_all_raw \
  --print-per-sequence-metrics
```

### 11.2 MOT20 all raw

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset MOT20 --mode all \
  --sequences MOT20-01 MOT20-02 MOT20-03 MOT20-05 \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_interleaved_shard1x20_seed42_bs1024/checkpoints/iwg_rg_cma_epoch100.pt" \
  --iwg-attn-output final \
  --tracker-suffix mot20_interleaved_e100_final_all_raw \
  --print-per-sequence-metrics
```

### 11.3 SportsMOT val raw

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset SportsMOT --mode val \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_sportsmot_trainval_seed42_bs1024_interleaved_shard1x20_200e/checkpoints/iwg_rg_cma_epoch200.pt" \
  --iwg-attn-output final \
  --tracker-suffix sportsmot_trainval_e200_final_val_raw \
  --print-per-sequence-metrics
```

如果权重用 SportsMOT train+val 训练，这条命令仍能运行，但 val 已参与训练，只能看拟合情况，不能用来声称泛化提升。

### 11.4 同一 checkpoint 比较 base 与 final

不要复制两份完整命令。可以在 validation 命令外套循环，只切换 gate 和 suffix：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
for GATE in base final; do
  "../.venv/bin/python" -u run.py \
    --dataset MOT20 --mode all \
    --sequences MOT20-01 MOT20-02 MOT20-03 MOT20-05 \
    --agentguard-mode iwg-attn \
    --agentguard-checkpoint \
      "../outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_interleaved_shard1x20_seed42_bs1024/checkpoints/iwg_rg_cma_epoch100.pt" \
    --iwg-attn-output "$GATE" \
    --tracker-suffix "mot20_interleaved_e100_${GATE}_all_raw" \
    --print-per-sequence-metrics
done
```

运行循环前只需先执行一次 `cd`。

## 12. test：final + post

test 命令统一使用：

```text
--iwg-attn-output final
--use_post
--skip-eval
```

官方 test 没有本地 GT，因此命令不会打印 HOTA 等指标。它只生成提交 txt；HOTA 需要上传官方评测服务器后获得。

### 12.1 MOT17 test final + post

下面使用当前 MOT20 e050 到 MOT17 fine-tune 25e 权重作为真实示例：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset MOT17 --mode test \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_mot20e050_to_mot17_finetune25_cachefix_seed42_bs1024/checkpoints/iwg_rg_cma_epoch025.pt" \
  --iwg-attn-output final \
  --tracker-suffix mot20e050_mot17_finetune25 \
  --use_post --skip-eval
```

post 结果目录：

```text
outputs/3. track/mot17_test_0.80_mot20e050_mot17_finetune25_iwg_attn_final_post
```

### 12.2 MOT20 test final + post

下面使用旧分片方式 epoch50 权重作为真实示例：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset MOT20 --mode test \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024_shard20x2/checkpoints/iwg_rg_cma_epoch050.pt" \
  --iwg-attn-output final \
  --tracker-suffix mot20_old_e050 \
  --use_post --skip-eval
```

post 结果目录：

```text
outputs/3. track/mot20_test_0.80_mot20_old_e050_iwg_attn_final_post
```

### 12.3 SportsMOT test final + post

下面使用 SportsMOT train+val、新分片方式 200e 的 epoch200 权重：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset SportsMOT --mode test \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_sportsmot_trainval_seed42_bs1024_interleaved_shard1x20_200e/checkpoints/iwg_rg_cma_epoch200.pt" \
  --iwg-attn-output final \
  --tracker-suffix trainval_interleaved_e200 \
  --use_post --skip-eval
```

post 结果目录：

```text
outputs/3. track/sportsmot_test_0.80_trainval_interleaved_e200_iwg_attn_final_post
```

### 12.4 MOT20 权重迁移到 MOT17 test

这组实验用于比较跨数据集泛化，不重新训练，使用 MOT17 的 NSA Kalman Filter runtime 和 MOT20 NSA event 旧分片方式的 epoch50 checkpoint。

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset MOT17 --mode test \
  --kf-type nsa \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024_shard20x2/checkpoints/iwg_rg_cma_epoch050.pt" \
  --iwg-attn-output final \
  --tracker-suffix mot20_nsa_old_e050_to_mot17 \
  --use_post --skip-eval
```

结果目录分别是：

```text
outputs/3. track/mot17_test_0.80_mot20_nsa_old_e050_to_mot17_iwg_attn_final_post
```

## 13. 输出目录命名规则

`iwg-attn` 的默认形式：

```text
outputs/3. track/<检测前缀>_0.80_<tracker-suffix>_iwg_attn_<base|final>
```

加 `--use_post` 后，结果目录末尾增加 `_post`：

```text
outputs/3. track/<检测前缀>_0.80_<tracker-suffix>_iwg_attn_<base|final>_post
```

例如 `--tracker-suffix mot20_nsa_old_e050_to_mot17` 会得到：

```text
mot17_test_0.80_mot20_nsa_old_e050_to_mot17_iwg_attn_final_post
```

旧脚本若显式传 `--legacy-output-naming`，才会继续使用带 `agentguard_` 的历史形式。

常见检测前缀：

```text
mot17_all      mot17_test
mot20_all      mot20_test
sportsmot_val  sportsmot_test
```

## 14. MOT17 test 打包

MOT17 tracker 只实际运行 7 个 FRCNN test 序列。提交时需要复制成 DPM/FRCNN/SDP 共 21 个文件，并检查 10 列格式、帧范围和 ZIP 根目录结构。使用仓库脚本，不要手工复制：

```bash
set -euo pipefail
./.venv/bin/python scripts/agentguard/package_mot17_submission.py \
  --source-dir outputs/3. track/mot17_test_0.80_mot20e050_mot17_finetune25_iwg_attn_final_post \
  --data-root /home/shang/datasets/MOT17/test \
  --output-zip outputs/agentguard/experiments/iwg_rg_cma_mot20e050_to_mot17_finetune25_cachefix_seed42_bs1024/MOT17_test_final_post.zip \
  --manifest outputs/agentguard/experiments/iwg_rg_cma_mot20e050_to_mot17_finetune25_cachefix_seed42_bs1024/MOT17_test_final_post_manifest.json
```

脚本会拒绝以下错误：缺少文件、不是 10 列、帧号越界、不是正好 21 个文件或 ZIP 内多了一层目录。

MOT20 和 SportsMOT 没有对应的 detector 三份扩展步骤。打包前应至少确认 txt 数量、每行 10 列，并保证 ZIP 根目录直接是 txt 文件，不要把结果目录本身包进 ZIP。

通用的 txt 数量与 10 列格式检查：

```bash
find "outputs/3. track/替换为实际post结果目录" -maxdepth 1 -type f -name '*.txt' | sort | wc -l
find "outputs/3. track/替换为实际post结果目录" -maxdepth 1 -type f -name '*.txt' \
  -exec awk -F, 'NF != 10 {print FILENAME ":" FNR ": " NF " columns"; bad=1} END {exit bad}' {} +
```

官方 test 的预期 txt 数量是：MOT20 为 4，SportsMOT 为 150。MOT17 在运行打包脚本后应为 21。

确认格式和数量无误后，可把 txt 直接放在 ZIP 根目录：

```bash
mkdir -p outputs/agentguard/experiments/替换为实验名
(cd "outputs/3. track/替换为实际post结果目录" && \
  zip -j "../../agentguard/experiments/替换为实验名/submission_final_post.zip" ./*.txt)
unzip -Z1 "outputs/agentguard/experiments/替换为实验名/submission_final_post.zip" | sed -n '1,20p'
```

`unzip -Z1` 输出应全是 `sequence.txt`，不能出现 `目录名/sequence.txt`。

## 15. 当前常用 checkpoint 速查

以下是已经存在并实际用于 test 的权重路径，不代表所有未来实验中永远最佳：

| 数据集 | 说明 | checkpoint |
| --- | --- | --- |
| MOT17 | MOT20 old e050 warm-start 到 MOT17 25e | `outputs/agentguard/experiments/iwg_rg_cma_mot20e050_to_mot17_finetune25_cachefix_seed42_bs1024/checkpoints/iwg_rg_cma_epoch025.pt` |
| MOT20 | NSA 事件、旧分片方式、epoch50 | `outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024_shard20x2/checkpoints/iwg_rg_cma_epoch050.pt` |
| MOT20 | NSA 事件、新分片方式 `5×1×20`，epoch25/50/75/100 | `outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_interleaved_shard1x20_seed42_bs1024/checkpoints/iwg_rg_cma_epoch{025,050,075,100}.pt` |
| SportsMOT | train+val、新分片方式、epoch200 | `outputs/agentguard/experiments/iwg_rg_cma_v1_sportsmot_trainval_seed42_bs1024_interleaved_shard1x20_200e/checkpoints/iwg_rg_cma_epoch200.pt` |

MOT20 新分片方式的权重已经存在，不需要重新准备数据或重新训练。其配置是 `epochs=100`、`batch_size=1024`、`memory_shards=5`、`epochs_per_shard=1`、`shard_cycles=20`，等效完整数据访问 20 轮。epoch100 的 `final` 指标为：raw `HOTA=0.791406, MOTA=0.934669, IDF1=0.919926, DetA=0.806762, AssA=0.776905`；post `HOTA=0.794985, MOTA=0.938744, IDF1=0.921030, DetA=0.810640, AssA=0.780232`。

已提交的 MOT20 test ZIP `outputs/3. track/mot20_test_0.80_best_mot20_interleaved_e100_final_test_agentguard_iwg_attn_final_post/mot20_test_0.80_best_interleaved_e100_iwg_attn_final_post_66.12.zip` 明确标记为 `interleaved_e100`，所以你记得的 66.12 确实来自 epoch100。公平对比旧方式时，应使用同一新方式实验的 `checkpoints/iwg_rg_cma_epoch050.pt`。

选择权重时，不要因为文件名是 `last` 就默认它最好。应根据 raw validation 选择 epoch；train+val 最终刷榜没有独立 validation 时，才结合已完成的 test 结果选择。

## 16. 常见错误

### 16.1 `run.py` 找不到相对路径

原因通常是没有进入 tracker 目录。先执行：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
```

### 16.2 训练提示拒绝覆盖 checkpoint

错误类似：

```text
formal IWG-attn training requires a new checkpoint directory
```

这是保护机制。给 `RUN` 换一个新实验名，不要删除旧权重后在原目录上强行重跑。

### 16.3 分片公式报错

三个参数必须严格满足：

```text
memory_shards * epochs_per_shard * shard_cycles = epochs
```

例如 `5 * 1 * 20 = 100` 有效，`5 * 1 * 10 = 50` 不能配 `--epochs 100`。

### 16.4 validation 为什么没有 post 指标

这是当前实验规范，不是漏参数。validation/all 只比较 raw，避免后处理掩盖 tracker 和 gate 本身的差异。post 只用于最终 test 提交。

### 16.5 test 为什么不打印 HOTA

官方 test 没有本地 GT。`--skip-eval` 只生成结果，五项指标需要提交官方服务器。

### 16.6 为什么推理使用 CPU

IWG + RG-CMA 在 tracker 中按在线事件运行，单次前向很小。使用 GPU 会引入频繁的 CPU/GPU 数据传输和同步，实际总时间不一定更快。因此正式示例使用 `--agentguard-device cpu`。这与训练必须使用 CUDA 不矛盾。

### 16.7 怎么确认是 NSA runtime

`run.py` 默认 `--kf-type nsa`。本文命令没有显式传 `--kf-type`，因此输出开头应看到：

```text
Running <dataset> <mode> with nsa Kalman filter...
```

如果想让命令更显式，可额外加入 `--kf-type nsa`，行为不变。

## 17. 最短速查

### 17.1 不使用变量的最短 test 命令

`run.py` 当前默认值已经是 `seed=10000`、`kf-type=nsa`、`iwg-attn-output=final`、`agentguard-device=cpu`，并且从 `3. Tracker` 目录运行时会自动使用 `../outputs/agentguard/detection_cache`。因此 test 可以直接写成：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"

"../.venv/bin/python" -u run.py \
  --dataset MOT17 --mode test \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024/checkpoints/iwg_rg_cma_epoch050.pt" \
  --tracker-suffix mot20_nsa_old_e050_to_mot17 \
  --use_post --skip-eval
```

NSA 旧方式 epoch50 只需要替换 checkpoint 和 suffix：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"

"../.venv/bin/python" -u run.py \
  --dataset MOT17 --mode test \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024_shard20x2/checkpoints/iwg_rg_cma_epoch050.pt" \
  --tracker-suffix mot20_nsa_old_e050_to_mot17 \
  --use_post --skip-eval
```

这里没有设置任何环境变量；只要先执行 `cd .../3. Tracker`，相对 checkpoint 和默认 detection cache 就会生效。

只训练已有 dataset：

```bash
cd "/home/shang/workspace/TrackTrack"
./.venv/bin/python -u -m agentguard.cli train_iwg_attn \
  --dataset-dir outputs/agentguard/experiments/替换为数据目录/dataset \
  --checkpoint-dir outputs/agentguard/experiments/替换为新的实验名/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 4 \
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 \
  --memory-shards 5 --epochs-per-shard 1 --shard-cycles 20
```

跑 raw validation：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset MOT20 --mode all \
  --sequences MOT20-01 MOT20-02 MOT20-03 MOT20-05 \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_interleaved_shard1x20_seed42_bs1024/checkpoints/iwg_rg_cma_epoch100.pt" \
  --iwg-attn-output final \
  --tracker-suffix mot20_interleaved_e100_final_all_raw \
  --print-per-sequence-metrics
```

跑 test + post：

```bash
cd "/home/shang/workspace/TrackTrack/3. Tracker"
"../.venv/bin/python" -u run.py \
  --dataset MOT20 --mode test \
  --agentguard-mode iwg-attn \
  --agentguard-checkpoint \
    "../outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_interleaved_shard1x20_seed42_bs1024/checkpoints/iwg_rg_cma_epoch100.pt" \
  --iwg-attn-output final \
  --tracker-suffix mot20_interleaved_e100 \
  --use_post --skip-eval
```

每次真正运行前，最后检查四件事：dataset 路径是否正确、checkpoint 是否是目标 epoch、suffix 是否唯一、当前是在 validation raw 还是 test post。
