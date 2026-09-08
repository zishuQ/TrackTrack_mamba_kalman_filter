# MOT17 train_data_v2：纯 TrackTrack 采集 + GT/LOO

基于 `agentguard` 分支 `5f8c1c1a713bc82798a84d1424182b349121595e`。
本次只支持七个 MOT17 FRCNN 序列，不迁移旧 train_data，不删除旧数据。

## 合并

压缩包包含 `train_data_v2.patch` 和 `files/` 中的改动后文件。
在仓库根目录先执行 `git apply --check /解压目录/train_data_v2.patch`，通过后再执行
`git apply /解压目录/train_data_v2.patch`。本地存在重叠改动时，交给本地 Codex 按补丁合并，保留其他改动。
不要把 `files/` 整体覆盖仓库。此前的 `train_data_new` 重建/复用旧 ReID 方案不与本方案叠加。
本补丁包含可选 GT/LOO 的实现；如果本地已经合并过标签补丁，应合并重叠部分，不重复应用。

原有训练数据读取方式继续可用。旧标签入口默认仍是旧规则；新 v2 脚本明确启用 GT + leave_one_out。
`cache_events` 入口改为 `agentguard_mode="capture"`：最新 Tracker 的 `off` 不创建采集 adapter，
原入口的 off 配置会漏掉事件。capture 创建记录器，但匹配、KF 和外观更新走纯 TrackTrack，未加载门控 checkpoint。

## 先采集、构建一个序列

以下命令均在 `/home/shang/workspace/TrackTrack` 仓库根目录和你原本能运行 TrackTrack 的 Python 环境执行。

```bash
python scripts/agentguard/build_mot17_train_data_v2.py --sequence MOT17-09-FRCNN
```

默认输入和输出：

- detection 根目录：`outputs/agentguard/detection_cache`，脚本自行追加 `MOT17/all/序列`。
- GT：`/home/shang/datasets/MOT17/train/序列/gt/gt.txt`。
- seqinfo：`/home/shang/datasets/MOT17/train/序列/seqinfo.ini`。
- GMC：仓库现有 `3. Tracker/trackers/cmc/GMC-MOT17-09.txt` 等文件。
- 输出：`outputs/agentguard/train_data_v2/MOT17`。

脚本顺序执行：共享检测缓存 → 纯 TrackTrack（NSA KF、GMC 开启）→ 完整事件采集 →
GT/LOO 标签 → 打包 → 状态逐字段还原对照、文件校验 → 发布最终序列 → 清理本次临时分片。
不运行检测器或 ReID 网络。使用最新纯 TrackTrack 本次运行的轨迹特征，不比对或复用旧训练文件里的轨迹特征。

需自定义路径时：

```bash
python scripts/agentguard/build_mot17_train_data_v2.py \
  --sequence MOT17-09-FRCNN \
  --data-dir /home/shang/datasets \
  --detection-cache-root /home/shang/workspace/TrackTrack/outputs/agentguard/detection_cache \
  --output-root /home/shang/workspace/TrackTrack/outputs/agentguard/train_data_v2
```

`--detection-cache-root` 不要写到 `MOT17/all`；`--data-dir` 不要写到 `MOT17/train`。
采集日志在 `train_data_v2/MOT17/.work/序列/capture.log`，失败时保留，成功打包后清理。

一个序列完成后，去掉 `--sequence` 即处理七个 FRCNN 序列，并校验、跳过已完成的序列：

```bash
python scripts/agentguard/build_mot17_train_data_v2.py
```

默认 context_size=6、max_frame_gap=30、future_frames=5。
如原实验为 8 帧上下文，第一次构建就加 `--context-size 8`，训练参数也用 8。
同一个输出根目录不能混合这些配置。完成全部所需序列后再正式训练；新增序列会更新根目录的归一化统计。

## 最终保存什么

每个序列下只有以下三个正式文件（另有数据集级 metadata.json 和作业锁）：

| 文件 | 内容 |
|---|---|
| `data.pt` | 训练窗口、GT/LOO 标签及原始标签；全部采集事件；frame-start / pre-update KF 状态、协方差、近期历史；逐帧 GMC；关联索引和矩阵；外部资源引用及生成信息 |
| `reid_features.npy` | 本次采集的更新前轨迹 ReID，float32，形状 `(事件数, ReID维数)` |
| `manifest.json` | 完成标记、配置、资源指纹、文件校验值、标量统计 |

`data.pt` 中：`arrays` 是训练数组，`replay.tables` 是按列打包的事件/状态/帧/关联记录，
`labels_raw` 是原始标签，`metadata` 是 schema/version、引用等信息。
变长历史和关联矩阵按扁平数据 + offsets + shapes 保存；轨迹 ReID 与原始 63 维标量不在 replay 中再次复制。
训练时 mmap 打开 `data.pt`，只保留训练数组，不逐事件展开关联矩阵；原始标量按根 metadata 的统计归一化。

检测 ReID 通过 `arrays.timeline_detection_indices` 指向共享 detection cache 的 `features.npy`；
这是该序列检测缓存的全局行号，不是“帧内第几个检测”。失配为 -1，读取时检测特征补零。
新 `reid_features.npy` 不再是旧版 `(N, 2, D)`。在事件数与维数相同的条件下，该文件大小约减半；
`data.pt` 因保留运动和关联状态会变大，不能据此保证总目录缩小。

保留的是现有标签算法离线重算所需的全部采集内容，遵循原来成熟轨迹的采样规则。
它不是用于从任意视频帧恢复在线 Tracker 的运行 checkpoint。
GT、seqinfo 和 detection cache 都引用现有文件，完全不复制。删除旧 train_data 不会破坏新数据；
共享 detection cache 仍然是新数据的依赖，必须保留。训练本身不读取 GT；未来重算标签需要 GT。

## 开始训练

一序列可用于流程试跑，正式对照建议先完成全部七个序列。
训练入口不变，将 dataset-dir 指向新的 MOT17 根目录，并使用新的 checkpoint-dir。
以下使用仓库原默认的 legacy 架构、6 帧上下文、0.05 修正范围；与原实验保持一致才能比较标签效果。

```bash
env PYTHONPATH="$PWD/agentguard/src:$PWD/3. Tracker" \
python -m agentguard.cli train_iwg_rg_cma \
  --dataset-dir outputs/agentguard/train_data_v2/MOT17 \
  --checkpoint-dir outputs/agentguard/experiments/mot17_gt_loo_v2/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 4 \
  --lr 1e-4 --correction-bound 0.05 --context-size 6 \
  --architecture-variant legacy --seed 42
```

safe target 公式与默认参数不变。运动评价使用 GT 宽高归一化（当前帧和未来步均修正）。
LOO 按身份收集质量合格的唯一检测，先去掉当前检测，再重算初始均值、相似度最高的 70% 和最终均值。
排除后不足 3 个参考检测，当前外观监督无效，运动监督仍可保留。

## 中断、重算和搬家

- 再运行相同命令可复用完成序列、完整采集阶段和完整标签阶段。
- 采集到一半中断：仅该序列从头重跑 TrackTrack；不能拼接半段事件继续。
- 打标签到一半中断：该序列标签从头计算，完成的事件采集可复用。
- 打包中断：完成的标签复用，打包重新执行。只有完整校验、发布成功才清理本次 `.work/序列`。
- 输入、代码或配置变化会拒绝复用不相符的已完成/未完成目录，使用新输出目录。不会静默覆盖。
- `--event-cache-root` 可选，用于首次打包已有的**完整原始事件缓存**，结构为 root/MOT17/all/序列。
  这些显式传入的缓存从不删除；旧训练 `data.pt` 不能用作该参数。

以后重算（此时原始事件分片已无需保留）：

```bash
python scripts/agentguard/build_mot17_train_data_v2.py \
  --sequence MOT17-09-FRCNN \
  --relabel-from outputs/agentguard/train_data_v2 \
  --output-root outputs/agentguard/train_data_v2_relabel
```

这条命令完全从旧 v2 的 `data.pt` / 轨迹 ReID、共享 detection cache、GT 重新计算标签，不再跑 TrackTrack。
它产生独立的新目录和独立的轨迹特征文件，保留作为来源的 v2；无需为了调整标签覆盖原数据。
重算范围仍是已采集事件；若要换 Tracker 算法、匹配规则或重新采样事件，需重新采集。

引用默认是相对路径。将 `outputs/agentguard` 整体搬家可保持 detection 相对布局。
若单独移动 detection cache，训练时设置环境变量：

```bash
env AGENTGUARD_DETECTION_CACHE_ROOT=/新位置/detection_cache \
  PYTHONPATH="$PWD/agentguard/src:$PWD/3. Tracker" \
  python -m agentguard.cli train_iwg_rg_cma [你的其余训练参数]
```

上面的方括号是说明占位符，需替换。Python 读取器也支持 `detection_cache_root=...`。
重算时直接传新 `--detection-cache-root` 和 `--data-dir`。搬迁保留数据集根 metadata.json。
构建/重算完整校验所有共享缓存文件的 SHA256；训练初始化只校验 manifest/hash、文件大小及索引，
避免每个 worker 启动都扫描整个 ReID 文件。共享 detection cache 应视为不可变资源。

## 验证范围

相关回归结果：39 passed（CPU 环境的 pin_memory 提示不影响测试）。

使用合成完整序列验证了：旧读取器兼容；GT 当前/未来归一化；LOO 排除与最小参考数；
真实 CompactEventCacheSink 输出到 v2；所有 replay 字段还原；移除原始事件缓存与旧数据后的标签独立重算；
共享检测特征数值一致；缓存搬迁与损坏校验；失配事件无关联记录；标签阶段恢复；
CPU 一轮训练、保存和重新加载验证 checkpoint；追加第二个序列更新全局归一化且不重写第一个 data.pt。
采集 CLI 测试使用 Tracker 替身检查 capture/NSA/GMC 配置和真实 sink 写入。
本环境没有你本地完整 MOT17 缓存、GMC 文件和 GPU，尚未运行真实整序列 TrackTrack、正式 CUDA 训练，未测 HOTA/IDF1。

在完整仓库执行回归：

```bash
env PYTHONPATH="$PWD/agentguard/src:$PWD/3. Tracker" python -m pytest -q \
  agentguard/tests/test_train_data_v2.py \
  agentguard/tests/test_motion_rollout.py \
  agentguard/tests/test_appearance_rollout.py \
  agentguard/tests/test_rollout_label_builder.py \
  agentguard/tests/test_iwg_rg_cma_dataset.py
```
