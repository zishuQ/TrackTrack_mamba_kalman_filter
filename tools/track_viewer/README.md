# Track Viewer

本地结果对比浏览器（左右两栏），支持：
- 左右选择结果文件夹与序列
- 每次前后跳 n 帧
- 直接输入帧号跳转
- 左右同步切帧
- 叠加显示 track id / 置信度
- 点击图片查看大图并保存当前可视化 PNG
- 每栏按当前帧 ID 选择显示 / 隐藏哪些框（跨帧保留，切序列重置）

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
