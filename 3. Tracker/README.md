## Run
Tracking results will be created under "../outputs/3. track/"

```
# For MOT17 validation
python run.py --dataset "MOT17" --mode "val"

# For MOT17 test
python run.py --dataset "MOT17" --mode "test"
python gen_test_file.py

# For MOT20 validation
python run.py --dataset "MOT20" --mode "val"

# For MOT20 test
python run.py --dataset "MOT20" --mode "test"
python gen_test_file.py

# For DanceTrack validation
python run.py --dataset "DanceTrack" --mode "val"

# For DanceTrack test
python run.py --dataset "DanceTrack" --mode "test"

# Classical KF variants on the TrackTrack baseline path:
# nsa (default baseline) / kf / ekf / ukf
python run.py --dataset "SportsMOT" --mode "val" --kf-type "ukf"
```

## Alternative backbones (zero-intrusion to Mamba files)

You can run the alternative covariance models with a parallel entrypoint:

```
# Validation with any ALT backbone:
# lstm / gru / rnn / transformer / mlp / tcn / s4d
python run_alt.py --dataset "MOT17" --mode "val" --backbone "lstm" --alt_model_path <checkpoint_path>
```
