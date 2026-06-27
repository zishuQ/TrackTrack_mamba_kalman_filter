## Datasets
Save .json files under "./jsons/" 
  - [mot17_val.json](https://drive.google.com/file/d/1hqcoFTtdzd5xMrC_xgz6mniI_sKg_0G9/view?usp=drive_link)
  - [mot17_test.json](https://drive.google.com/file/d/1CQ91C7Hl4B2rDfy_IU2orusD9vaiD5cs/view?usp=drive_link)
  - [mot20_val.json](https://drive.google.com/file/d/16IrR-TWc-K6c6NHwjM3OV74NBXPrxR-_/view?usp=drive_link)
  - [mot20_test.json](https://drive.google.com/file/d/1h3EkjOpcn058g2tgGGEcD7r5sAGgskwg/view?usp=drive_link)
  - [dance_val.json](https://drive.google.com/file/d/1O__fCM3gPbzHtav3XrlzHjjs96Dl45m8/view?usp=drive_link)
  - [dance_test.json](https://drive.google.com/file/d/12rBCIYLCXqT8bYmNrEwNdp6MmJ7whEg-/view?usp=drive_link)

## Model Zoo
Save weights files under "./weights/"
  - [mot17_half.pth.tar](https://drive.google.com/file/d/1R-eMf5SgwmizMkOjqJq3ZiurWBNGYf1j/view?usp=drive_link)
  - [mot17.pth.tar](https://drive.google.com/file/d/1MAb-Bhikx-fWe0VlJON_VMrYIyyyrt-F/view?usp=drive_link)
  - [mot20_half.pth.tar](https://drive.google.com/file/d/1H1BxOfinONCSdQKnjGq0XlRxVUo_4M8o/view?usp=drive_link)
  - [mot20.pth.tar](https://drive.google.com/file/d/1FunATdHrWfK95RiiEIw2GJ-gXB-tXMPB/view?usp=drive_link)
  - [dance.pth.tar](https://drive.google.com/file/d/1ZKpYmFYCsRdXuOL60NRuc7VXAFYRskXB/view?usp=drive_link)

### One-command auto download
Install dependency (if not already):
```
pip install gdown
```
Then run inside this folder:
```
python download_assets.py --with-dets
```
Options:
```
python download_assets.py                 # json + weights
python download_assets.py --only json     # only json
python download_assets.py --only weights  # only weights
python download_assets.py --with-dets     # plus detection result pickles
```
Existing files are skipped.

## Detection Results
Save detection result files under "../outputs/1. det/"
  - [detection result files](https://drive.google.com/drive/folders/1Ef-O0DCZAS8ObqJ9cv751ils-KehSgA7?usp=sharing)
    
## Run
Detection results will be created under "../outputs/1. det/" as pickle files
```
# For MOT17 validation
python detect.py -f "exps/yolox_x_mot17_val.py" -c "weights/mot17_half.pth.tar" --nms 0.80 -n "../outputs/1. det/mot17_val_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_mot17_val.py" -c "weights/mot17_half.pth.tar" --nms 0.95 -n "../outputs/1. det/mot17_val_0.95.pickle" -b 1 -d 1 --fp16 --fuse

# For MOT17 test
python detect.py -f "exps/yolox_x_mot17_test.py" -c "weights/mot17.pth.tar" --nms 0.80 -n "../outputs/1. det/mot17_test_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_mot17_test.py" -c "weights/mot17.pth.tar" --nms 0.95 -n "../outputs/1. det/mot17_test_0.95.pickle" -b 1 -d 1 --fp16 --fuse

# For MOT20 validation
python detect.py -f "exps/yolox_x_mot20_val.py" -c "weights/mot20_half.pth.tar" --nms 0.80 -n "../outputs/1. det/mot20_val_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_mot20_val.py" -c "weights/mot20_half.pth.tar" --nms 0.95 -n "../outputs/1. det/mot20_val_0.95.pickle" -b 1 -d 1 --fp16 --fuse

# For MOT20 test
python detect.py -f "exps/yolox_x_mot20_test.py" -c "weights/mot20.pth.tar" --nms 0.80 -n "../outputs/1. det/mot20_test_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_mot20_test.py" -c "weights/mot20.pth.tar" --nms 0.95 -n "../outputs/1. det/mot20_test_0.95.pickle" -b 1 -d 1 --fp16 --fuse

# For DanceTrack val
python detect.py -f "exps/yolox_x_dance_val.py" -c "weights/dance.pth.tar" --nms 0.80 -n "../outputs/1. det/dance_val_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_dance_val.py" -c "weights/dance.pth.tar" --nms 0.95 -n "../outputs/1. det/dance_val_0.95.pickle" -b 1 -d 1 --fp16 --fuse

# For DanceTrack test
python detect.py -f "exps/yolox_x_dancetrack_test.py" -c "weights/dancetrack.pth.tar" --nms 0.80 -n "../outputs/1. det/dance_test_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_dancetrack_test.py" -c "weights/dancetrack.pth.tar" --nms 0.95 -n "../outputs/1. det/dance_test_0.95.pickle" -b 1 -d 1 --fp16 --fuse

# For SportsMOT val
python detect.py -f "exps/yolox_x_sportsmot_val.py" -c "weights/SportsMOT_yolox_x_mix.tar" --nms 0.80 -n "../outputs/1. det/sportsmot_val_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_sportsmot_val.py" -c "weights/SportsMOT_yolox_x_mix.tar" --nms 0.95 -n "../outputs/1. det/sportsmot_val_0.95.pickle" -b 1 -d 1 --fp16 --fuse

# For SportsMOT train
/home/shang/workspace/TrackTrack/.venv/bin/python detect.py -f "exps/yolox_x_sportsmot_train.py" -c "weights/SportsMOT_yolox_x_mix.tar" --nms 0.80 -n "../outputs/1. det/sportsmot_train_0.80.pickle" -b 1 -d 1 --fp16 --fuse
/home/shang/workspace/TrackTrack/.venv/bin/python detect.py -f "exps/yolox_x_sportsmot_train.py" -c "weights/SportsMOT_yolox_x_mix.tar" --nms 0.95 -n "../outputs/1. det/sportsmot_train_0.95.pickle" -b 1 -d 1 --fp16 --fuse

# For SportsMOT test
python detect.py -f "exps/yolox_x_sportsmot_test.py" -c "weights/SportsMOT_yolox_x_mix.tar" --nms 0.80 -n "../outputs/1. det/sportsmot_test_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_sportsmot_test.py" -c "weights/SportsMOT_yolox_x_mix.tar" --nms 0.95 -n "../outputs/1. det/sportsmot_test_0.95.pickle" -b 1 -d 1 --fp16 --fuse

```

## Reference
  - https://github.com/Megvii-BaseDetection/YOLOX
  - https://github.com/ifzhang/ByteTrack
