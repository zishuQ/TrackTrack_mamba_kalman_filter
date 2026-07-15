import argparse
from utils.etc import evaluate, evaluate_sequences

# Parse arguments
parser = argparse.ArgumentParser("Evaluate tracking results")
parser.add_argument("--tracker_name", type=str, required=True, help="Tracker folder name")
parser.add_argument("--dataset", type=str, default="MOT20", help="Dataset name")
parser.add_argument("--mode", type=str, default="train_custom", help="Split mode")
parser.add_argument("--data_dir", type=str, default="/home/shang/datasets/")
parser.add_argument("--output_dir", type=str, default="../outputs/3. track/")
parser.add_argument(
    "--sequences",
    type=str,
    nargs="+",
    default=None,
    help="Only evaluate specific sequences. Supports space-separated or comma-separated names.",
)
parser.add_argument(
    "--print-per-sequence-metrics",
    action="store_true",
    help="Print HOTA/MOTA/IDF1/DetA/AssA for each evaluated sequence.",
)
parser.add_argument("--no_post", action="store_true", help="Evaluate raw tracker folder instead of <tracker_name>_post")

args = parser.parse_args()
if args.sequences:
    args.sequences = [
        seq.strip()
        for item in args.sequences
        for seq in item.split(',')
        if seq.strip()
    ]

# Set data path based on mode
if 'MOT17' in args.dataset:
    if args.mode in ['val', 'val_custom', 'train_custom', 'all']:
        args.data_path = args.data_dir + 'MOT17/train/'
    else:
        args.data_path = args.data_dir + 'MOT17/test/'
elif 'MOT20' in args.dataset:
    if args.mode in ['val', 'val_custom', 'train_custom', 'all']:
        args.data_path = args.data_dir + 'MOT20/train/'
    else:
        args.data_path = args.data_dir + 'MOT20/test/'
elif 'sportsmot' in args.dataset.lower():
    args.data_path = args.data_dir + 'SportsMOT/dataset/' + (
        'val/' if args.mode == 'val' else 'test/'
    )
else:
    args.data_path = args.data_dir + 'DanceTrack/' + ('val/' if args.mode == 'val' else 'test/')

# Run evaluation
print(f'Evaluating {args.tracker_name} on {args.dataset} {args.mode}...')
tracker_name = args.tracker_name if args.no_post else args.tracker_name + '_post'
if args.sequences:
    evaluate_sequences(args, tracker_name, args.dataset, args.sequences)
else:
    evaluate(args, tracker_name, args.dataset)
