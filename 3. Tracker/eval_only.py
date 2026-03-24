import argparse
from utils.etc import evaluate

# Parse arguments
parser = argparse.ArgumentParser("Evaluate tracking results")
parser.add_argument("--tracker_name", type=str, required=True, help="Tracker folder name")
parser.add_argument("--dataset", type=str, default="MOT20", help="Dataset name")
parser.add_argument("--mode", type=str, default="train_custom", help="Split mode")
parser.add_argument("--data_dir", type=str, default="/home/shang/datasets/")
parser.add_argument("--output_dir", type=str, default="../outputs/3. track/")

args = parser.parse_args()

# Set data path based on mode
if 'MOT17' in args.dataset:
    if args.mode in ['val', 'val_custom', 'train_custom']:
        args.data_path = args.data_dir + 'MOT17/train/'
    else:
        args.data_path = args.data_dir + 'MOT17/test/'
elif 'MOT20' in args.dataset:
    if args.mode in ['val', 'val_custom', 'train_custom']:
        args.data_path = args.data_dir + 'MOT20/train/'
    else:
        args.data_path = args.data_dir + 'MOT20/test/'
else:
    args.data_path = args.data_dir + 'DanceTrack/' + ('val/' if args.mode == 'val' else 'test/')

# Run evaluation
print(f'Evaluating {args.tracker_name} on {args.dataset} {args.mode}...')
evaluate(args, args.tracker_name+'_post', args.dataset)
