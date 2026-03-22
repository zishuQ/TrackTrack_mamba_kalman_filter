# Track Viewer

本地结果对比浏览器（左右两栏），支持：
- 左右选择结果文件夹与序列
- 每次前后跳 n 帧
- 直接输入帧号跳转
- 左右同步切帧
- 叠加显示 track id

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
