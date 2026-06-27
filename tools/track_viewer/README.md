# Track Viewer

本地结果对比浏览器（左右两栏），支持：
- 左右选择结果文件夹与序列
- 每次前后跳 n 帧
- 直接输入帧号跳转
- 左右同步切帧
- 叠加显示 track id / 置信度
- 可选叠加显示 GT ID（有 GT 的序列）
- 点击图片查看大图并保存当前可视化 PNG
- 左右两栏分别独立切换 `Track ID / GT ID` 过滤模式
- 每栏按当前帧的 Track ID / GT ID 选择显示 / 隐藏哪些框（跨帧保留；切序列时保留该栏模式，但重置该栏显隐集合）
- GT 过滤模式下会把未匹配任何 GT 的框单独归为 `未匹配`

## 运行

在项目根目录执行（fish）：

```fish
source /home/shang/workspace/TrackTrack/.venv/bin/activate.fish
python /home/shang/workspace/TrackTrack/tools/track_viewer/app.py
```

然后在浏览器打开：

```
http://127.0.0.1:5000
```

## 路径配置

默认数据集根目录：
- DanceTrack: `~/datasets/DanceTrack`
- MOT17: `~/datasets/MOT17`
- MOT20: `~/datasets/MOT20`
- SportsMOT: `~/datasets/SportsMOT/dataset`

可用环境变量覆盖：
- `TRACKVIEW_DANCETRACK_ROOT`
- `TRACKVIEW_MOT17_ROOT`
- `TRACKVIEW_MOT20_ROOT`
- `TRACKVIEW_SPORTSMOT_ROOT`

## 数据集 split 说明

- `MOT17` / `MOT20` 的结果文件夹即使名字里包含 `val`、`all`、`val_custom`、`train_custom`，viewer 也会按仓库现有约定去物理 `train/` 目录下找图片和 GT
- 只有明确带 `test` 的 `MOT17` / `MOT20` 结果文件夹才会去物理 `test/` 目录
- `SportsMOT` 仍按 `train` / `val` / `test` 物理目录解析

## GT 模式说明

- 当前 GT 匹配规则与参考脚本 `draw_mot_comparison.py::match_gt_tracker()` 保持一致：
  - 逐帧匹配
  - 仅保留 `IoU > 0.3` 的 GT / tracker 框对
  - 按 IoU 从高到低做贪心一对一分配
- 当前支持 GT 的常见序列：
  - `MOT17` / `MOT20` 的 train 派生模式（如 `val`、`all`、`*_custom`）
  - `SportsMOT` 的 `train` / `val`
- 切换到 `GT ID` 过滤模式后：
  - chips 会显示当前帧里匹配到的 GT ID
  - 额外提供 `未匹配` 项，用于显隐没有匹配到 GT 的 tracker 框
  - 如果当前序列没有 GT，会自动退回 Track ID 过滤
- 过滤模式是**按栏独立**的：
  - 左侧可以保持 `Track ID`
  - 右侧同时切到 `GT ID`
  - 切换序列后会保留该栏当前模式，方便跨序列持续对照

## 推荐工作流（SportsMOT val 对比）

1. 先生成每序列 HOTA 差值排序：

```fish
cd /home/shang/workspace/TrackTrack
./.venv/bin/python tools/repro_harness/trackeval_compare.py \
  --dataset SportsMOT \
  --mode val \
  --improved-tracker sportsmot_val_0.80_mamba_post_83.92_ablation_baseline \
  --baseline-tracker sportsmot_val_0.80_mamba
```

2. 打开浏览器后，左右文件夹分别选择：

- 左侧：`sportsmot_val_0.80_mamba_post_83.92_ablation_baseline`
- 右侧：`sportsmot_val_0.80_mamba`

3. 按 `per_sequence_delta.csv` 里 `delta_hota` 从大到小，优先检查排名靠前的序列。

4. 浏览时建议打开：

- `左右同步`
- `序列跟随`
- `仅看差异`
- `扫描差异帧`

5. 如果需要导出当前可视化结果：

- 先按当前帧 ID 过滤掉不想展示的框
- 再直接点击对应栏里的图片放大
- 在弹层里点 `保存 PNG`

这样可以更快定位提升最明显的帧段。
