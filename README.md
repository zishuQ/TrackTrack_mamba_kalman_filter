# TrackTrack
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focusing-on-tracks-for-online-multi-object/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=focusing-on-tracks-for-online-multi-object)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focusing-on-tracks-for-online-multi-object/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=focusing-on-tracks-for-online-multi-object)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focusing-on-tracks-for-online-multi-object/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=focusing-on-tracks-for-online-multi-object)<br>

Official code for "Focusing on Tracks for Online Multi-Object Tracking", CVPR, 2025
  - https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html

## Environment
Developed in python3.8, pytorch 1.13


## Prepare
**1. Downlodad datasets**
  - MOT17: https://motchallenge.net/data/MOT17.zip
  - MOT20: https://motchallenge.net/data/MOT20.zip
  - DanceTrack: https://dancetrack.github.io/

<br />

**2. Locate codes and datasets as below**
```
- workspace
  - code
    - 1. YOLOX
    - 2. FastReID
    - 3. Tracker
  - dataset
    - MOT17
    - MOT20
    - DanceTrack
```

<br />

**3. Run**
```
run 1. YOLOX
run 2. FastReID
run 3. Tracker
```

### Compact detection-feature storage

Tracker inputs now use the compact cache layout below:

```
outputs/2. det_feat/<prefix>_0.95.pickle
outputs/2. det_feat/<prefix>_0.80.from_<prefix>_0.95.idx.pickle
```

The tracker only loads the `0.95.pickle` cache and reconstructs the logical
`0.80` detection view by indexing rows with the `.idx.pickle` file. A physical
`<prefix>_0.80.pickle` file is not required.

Build or refresh the idx files from existing `0.95.pickle` caches:

```bash
python scripts/build_nms_idx_from_95.py --overwrite
```

Or process a single prefix:

```bash
python scripts/build_nms_idx_from_95.py --prefix mot17_val --overwrite
```

Run trackers as usual from `3. Tracker`; the output folder keeps the logical
`0.80` name, but the loaded cache is `0.95 + idx`:

```bash
cd "3. Tracker"
../.venv/bin/python run.py --dataset MOT17 --mode val
../.venv/bin/python run_mamba.py --dataset MOT17 --mode test --mamba_model_path /path/to/model.pth
```

## Results
<img src="https://github.com/user-attachments/assets/35063890-6684-4909-8215-e277cf20a1ac" width="550" height="550" />
<img src="https://github.com/user-attachments/assets/f3467ebe-5d6c-4179-9885-232ac2dfa07a" width="550" height="550" />
<img src="https://github.com/user-attachments/assets/5f389a10-a587-4b71-b277-c5830fc81dbb" width="550" height="550" />
