# Copilot Instructions for TrackTrack

## Build, test, and lint commands

Use the repository venv (`.venv`) and quote folder names with spaces.

### Build

- Build FastReID Cython evaluation extensions:
  - `cd '2. FastReID/fastreid/evaluation/rank_cylib' && ../../../../.venv/bin/python setup.py build_ext --inplace`

### Tests / checks

- Run a single evaluation test (Cython vs Python parity/speed):
  - `cd '2. FastReID/fastreid/evaluation/rank_cylib' && ../../../../.venv/bin/python test_cython.py`

- Syntax check used by this repo’s existing Mamba instructions:
  - Full core check:
    - `cd mamba_kalman_filter && ../.venv/bin/python -m py_compile config.py dataset.py mamba_kalmanfilter.py train.py resume_train.py trainer/*.py`
  - Single-file check:
    - `cd mamba_kalman_filter && ../.venv/bin/python -m py_compile mamba_kalmanfilter.py`

### End-to-end pipeline commands

- Stage 1 (detection pickle):
  - `cd '1. YOLOX' && ../.venv/bin/python detect.py -f exps/yolox_x_mot17_val.py -c weights/mot17_half.pth.tar --nms 0.80 -n '../outputs/1. det/mot17_val_0.80.pickle' -b 1 -d 1 --fp16 --fuse`

- Stage 2 (append ReID embeddings into detection pickle):
  - `cd '2. FastReID' && ../.venv/bin/python ext_feats.py --data_path '/home/shang/datasets/MOT17/train/' --pickle_path '../outputs/1. det/mot17_val_0.80.pickle' --output_path '../outputs/2. det_feat/mot17_val_0.80.pickle' --config_path 'configs/MOT17_half/sbs_S50.yml' --weight_path 'weights/mot17_half_sbs_S50.pth'`

- Stage 3 (tracking + evaluation on non-test modes):
  - `cd '3. Tracker' && ../.venv/bin/python run.py --dataset MOT17 --mode val`

- Mamba tracker variant:
  - `cd '3. Tracker' && ../.venv/bin/python run_mamba.py --dataset MOT17 --mode val --mamba_model_path ../mamba_kalman_filter/checkpoints/MOT20_best_model.pth.exp15_2`

### Lint

- There is no repository-level lint configuration/command (no root `ruff/flake8/pylint` config).

## High-level architecture

TrackTrack is a 3-stage MOT pipeline wired by pickle artifacts, plus an optional learned Kalman branch:

1. `1. YOLOX/detect.py` runs detector inference and writes nested dict pickles:
   - `det_results[video_name][frame_id] -> ndarray | None`
   - ndarray format from detector stage: `[x1, y1, x2, y2, score, class_id]`

2. `2. FastReID/ext_feats.py` loads detector pickles, crops boxes from frames, computes embeddings, and concatenates features:
   - output detection rows become `[x1, y1, x2, y2, score, class_id, reid_feat...]`

3. `3. Tracker/run.py` (or `run_mamba.py`) loads two detection-feature pickles (`0.80` and `0.95` NMS variants), runs association/tracking, writes MOTChallenge txt files, then optionally runs dataset-specific post-processing and TrackEval metrics.

Optional learned motion model:

- `mamba_kalman_filter/` trains `MambaKalmanFilter` and can be consumed by `3. Tracker/trackers/mamba_kalman_filter_wrapper.py`.
- `mamba_kalman_filter/train.py` can call tracker-side evaluation (`run_mamba.py`) for HOTA-based model selection.

## Key repository conventions

- Directory names contain spaces (`1. YOLOX`, `2. FastReID`, `3. Tracker`); always quote paths in shell commands.

- File naming is part of runtime wiring. `set_parameters()` in `3. Tracker/utils/etc.py` expects exact pickle names like:
  - `mot17_val_0.80.pickle` and `mot17_val_0.95.pickle`
  - Similar naming for MOT20/DanceTrack/SportsMOT and `val_custom/train_custom/all`.
  Renaming outputs requires updating `set_parameters()` (and usually corresponding seqmap/eval assumptions).

- Tracker logic relies on dual-threshold detections:
  - `0.80` file = primary detections
  - `0.95` file = deleted/high-confidence recovery candidates (`find_deleted_detections` path).

- Dataset-specific post-processing is intentional in `run.py`/`run_mamba.py`:
  - DanceTrack -> AFLink
  - MOT17/MOT20 -> Gaussian-based interpolation (`gb_interpolation`)
  - SportsMOT -> linear interpolation only

- Evaluation split semantics are fixed by TrackEval seqmaps under `3. Tracker/trackeval/seqmap/**`; preserve mode names (`val`, `val_custom`, `train_custom`, `all`, `test`) when adding flows.

- Tracker output folder structure is dataset-dependent; DanceTrack test uses a `tracker/` subfolder under the run directory.

- Mamba integration contracts (from existing local instructions in `mamba_kalman_filter/.github`):
  - Keep train/inference timing aligned (`last_obs` usage and GMC timing).
  - Keep wrapper normalization consistent (pixel <-> normalized coordinates, image size must be set per sequence).
  - Preserve numerically stable update paths (Cholesky solve, Joseph-form covariance update, positive covariance parameterization).

