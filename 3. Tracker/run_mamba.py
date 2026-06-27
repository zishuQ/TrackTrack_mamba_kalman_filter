import os
import sys
import torch
import pickle
import argparse
from tqdm import tqdm

# Add parent directory to path to access mamba_kalman_filter module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mamba_kalman_filter.config import Config
from utils.etc import *
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackers.tracker_mamba import TrackerMamba
from trackers.mamba_kalman_filter_wrapper import MambaKalmanFilterWrapper
from utils.gbi import gb_interpolation, linear_interpolation_only


def inject_detection_noise(detections_frame, noise_pos_std=0.0, noise_size_std=0.0, drop_rate=0.0):
    """Inject noise into single-frame detections for robustness testing.

    Args:
        detections_frame: numpy array (N, 2054), columns 0-3 are x1,y1,x2,y2
        noise_pos_std: std of Gaussian noise added to (x,y) position (pixels)
        noise_size_std: std of Gaussian noise added to (w,h) size (pixels),
                        applied symmetrically around bbox center
        drop_rate: probability of dropping each detection
    Returns:
        Modified detection array (may have fewer rows if drop_rate > 0)
    """
    if detections_frame is None or len(detections_frame) == 0:
        return detections_frame

    det = detections_frame.copy()

    # Position noise: shift (x1,x2) by same dx, (y1,y2) by same dy
    if noise_pos_std > 0:
        n = len(det)
        dx = np.random.normal(0, noise_pos_std, size=n)
        dy = np.random.normal(0, noise_pos_std, size=n)
        det[:, 0] += dx  # x1
        det[:, 2] += dx  # x2
        det[:, 1] += dy  # y1
        det[:, 3] += dy  # y2

    # Size noise: perturb width/height symmetrically around center
    if noise_size_std > 0:
        n = len(det)
        dw = np.random.normal(0, noise_size_std, size=n)
        dh = np.random.normal(0, noise_size_std, size=n)
        det[:, 0] -= dw / 2  # x1 shrinks/grows left
        det[:, 2] += dw / 2  # x2 shrinks/grows right
        det[:, 1] -= dh / 2  # y1 shrinks/grows up
        det[:, 3] += dh / 2  # y2 shrinks/grows down

    # Detection dropout
    if drop_rate > 0 and len(det) > 0:
        keep_mask = np.random.random(len(det)) >= drop_rate
        if keep_mask.sum() == 0:
            keep_mask[np.random.randint(len(det))] = True
        det = det[keep_mask]

    return det


def _normalize_config_dataset_name(dataset_name: str) -> str:
    """Normalize tracker dataset name to Config.load_dataset_config() naming."""
    lowered = dataset_name.lower()
    if lowered == 'sportsmot':
        return 'SPORTSMOT'
    if lowered == 'dancetrack':
        return 'DANCETRACK'
    if lowered == 'mot17':
        return 'MOT17'
    if lowered == 'mot20':
        return 'MOT20'
    if lowered == 'both':
        return 'BOTH'
    return dataset_name


def make_parser():
    parser = argparse.ArgumentParser("Tracker with MambaKalmanFilter")

    # Basic
    parser.add_argument("--pickle_dir", type=str, default="../outputs/2. det_feat/")
    parser.add_argument("--output_dir", type=str, default="../outputs/3. track/")
    parser.add_argument("--data_dir", type=str, default="/home/shang/datasets/")
    parser.add_argument("--dataset", type=str, default="MOT17")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--use_post", action="store_true", help="Use post-processing and evaluate post-processed results")
    parser.add_argument(
        "--print-per-sequence-metrics",
        action="store_true",
        help="Print HOTA/MOTA/IDF1/DetA/AssA for each evaluated sequence.",
    )

    # For trackers
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--max_time_lost", type=float, default=30)
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)
    
    # Mamba Kalman Filter
    parser.add_argument("--mamba_model_path", type=str, default=None, 
                       help="Path to MambaKalmanFilter model weights. If None, auto-detect based on dataset.")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA residual augmentation inside MambaCovNet")
    parser.add_argument("--no-mamba", action="store_true", help="Disable the Mamba sequence layer inside MambaCovNet")
    parser.add_argument("--enable-innov", action="store_true", help="Force-enable innovation feature in Q/R inputs")
    parser.add_argument("--disable-innov", action="store_true", help="Disable innovation feature in Q/R inputs")
    parser.add_argument("--enable-diou", action="store_true", help="Force-enable the 1D IoU-like feature branch in Q/R inputs")
    parser.add_argument("--disable-diou", action="store_true", help="Disable the 1D IoU-like feature branch in Q/R inputs")
    parser.add_argument(
        "--iou-feature-type",
        type=str,
        default="diou",
        choices=["diou", "iou", "giou", "ciou", "hmiou"],
        help="IoU-family variant used for the 1D IoU-like feature branch when enabled",
    )
    parser.add_argument("--zero-mask-innov", action="store_true", help="Keep innovation dim but zero innovation values")
    parser.add_argument("--zero-mask-diou", action="store_true", help="Keep the IoU-like feature dim but zero its values")
    parser.add_argument("--no-reid", action="store_true", help="Disable ReID (cosine distance) in association, use IoU-only")

    # Robustness test: noise injection
    parser.add_argument("--noise-pos-std", type=float, default=0.0,
                       help="Gaussian std (pixels) to add to detection (x,y) positions")
    parser.add_argument("--noise-size-std", type=float, default=0.0,
                       help="Gaussian std (pixels) to add to detection (w,h) size")
    parser.add_argument("--drop-rate", type=float, default=0.0,
                       help="Probability of dropping each detection (0.0=no drop)")
    parser.add_argument("--disable-gmc", action="store_true",
                       help="Disable Global Motion Compensation for robustness testing")
    parser.add_argument("--tracker-suffix", type=str, default="",
                       help="Suffix appended to tracker output folder name for unique results")
    parser.add_argument("--sequences", type=str, nargs="+", default=None,
                       help="Only track specific sequences (e.g. --sequences MOT20-01)")
    parser.add_argument("--num-trajectory-tokens", type=int, default=None,
                       help="Override NUM_TRAJECTORY_TOKENS in Config (for hyperparameter ablation)")
    parser.add_argument("--window-size", type=int, default=None,
                       help="Override WINDOW_SIZE and TTA_BUFFER_SIZE in Config (for hyperparameter ablation)")
    return parser


def track(detections, detections_95, data_path, result_folder, mode):
    # For each video
    total_time, total_count = 0, 0
    vid_names = list(detections.keys())
    for vid_name in tqdm(vid_names, desc="Processing videos", unit="video"):
        # Set proper parameters
        set_parameters(args, vid_name, mode)

        # Set max time lost
        with open(data_path + vid_name + '/seqinfo.ini', mode='r') as seq_info:
            for s_i in seq_info.readlines():
                if 'frameRate' in s_i:
                    args.max_time_lost = int(s_i.split('=')[-1]) * 2
                if 'imWidth' in s_i:
                    args.img_w = int(s_i.split('=')[-1])
                if 'imHeight' in s_i:
                    args.img_h = int(s_i.split('=')[-1])

        # Set tracker
        tracker = TrackerMamba(args, vid_name)
        
        # Initialize MambaKalmanFilter (shared instance for all tracks)
        model_path = args.mamba_model_path
        if model_path is None:
            # Auto-detect model path based on dataset
            if 'MOT20' in args.dataset:
                model_path = '../mamba_kalman_filter/checkpoints/MOT20_checkpoint_epoch_50.pth.exp15_1'
            elif 'MOT17' in args.dataset:
                model_path = '../mamba_kalman_filter/checkpoints/MOT20_best_model.pth.exp15_2'
            elif 'Dance' in args.dataset or 'dancetrack' in args.dataset.lower():
                model_path = '../mamba_kalman_filter/checkpoints/DANCETRACK_best_model.pth.exp1_1'
            elif 'Sports' in args.dataset or 'sports' in args.dataset.lower():
                model_path = '../mamba_kalman_filter/checkpoints/SPORTSMOT_best_model.pth.exp7_1'
        
        tracker.shared_kalman_filter = MambaKalmanFilterWrapper.get_shared_instance(
            model_path=model_path, device='cuda'
        )
        
        # Set image size for normalization (different videos may have different sizes)
        tracker.shared_kalman_filter.set_image_size(args.img_w, args.img_h)

        # For each frame
        results = []
        frame_ids = sorted(detections[vid_name].keys())
        for frame_id in tqdm(frame_ids, desc=f"  {vid_name}", leave=False, unit="frame"):
            # Run tracking
            start = time.time()
            if detections[vid_name][frame_id] is not None:
                det_frame = inject_detection_noise(
                    detections[vid_name][frame_id],
                    noise_pos_std=args.noise_pos_std,
                    noise_size_std=args.noise_size_std,
                    drop_rate=args.drop_rate,
                )
                det_frame_95 = detections_95[vid_name][frame_id]
                track_results = tracker.update(det_frame, det_frame_95)
            else:
                track_results = tracker.update_without_detections()
            total_time += time.time() - start
            total_count += 1

            # Filter out the results
            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                # Check aspect ratio
                if 'MOT' in data_path and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                    continue

                # Check track id, minimum box area
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)

            # Merge
            results.append([frame_id, track_ids, x1y1whs, scores])

        # Logging & Write results
        result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
        write_results(result_filename, results)
        
        # Clean up all remaining tracks' hidden states after video ends
        for track in tracker.tracks:
            tracker.shared_kalman_filter.delete_track(track.track_id)

    return total_time, total_count


def run():
    # Sync tracker-side feature flags with training-side config before model init
    Config.load_dataset_config(_normalize_config_dataset_name(args.dataset))
    if args.no_tta:
        Config.USE_TTA = False
    if args.no_mamba:
        Config.USE_MAMBA = False
    if args.enable_innov and args.disable_innov:
        raise ValueError("--enable-innov and --disable-innov cannot be used together")
    if args.enable_diou and args.disable_diou:
        raise ValueError("--enable-diou and --disable-diou cannot be used together")

    if args.enable_innov:
        Config.USE_INNOVATION_FEATURE = True
    elif args.disable_innov:
        Config.USE_INNOVATION_FEATURE = False
    if args.enable_diou:
        Config.USE_DIOU_FEATURE = True
    elif args.disable_diou:
        Config.USE_DIOU_FEATURE = False
    Config.IOU_FEATURE_TYPE = args.iou_feature_type.lower()
    Config.ZERO_MASK_INNOVATION = bool(args.zero_mask_innov)
    Config.ZERO_MASK_DIOU = bool(args.zero_mask_diou)
    if args.num_trajectory_tokens is not None:
        Config.NUM_TRAJECTORY_TOKENS = args.num_trajectory_tokens
    if args.window_size is not None:
        Config.WINDOW_SIZE = args.window_size
        Config.TTA_BUFFER_SIZE = args.window_size

    # Logging & Set proper parameters
    print('Running %s %s with MambaKalmanFilter...' % (args.dataset, args.mode))
    print(
        f"Model flags: use_tta={Config.USE_TTA}, use_mamba={Config.USE_MAMBA}"
    )
    print(
        f"Feature flags: use_innov={Config.USE_INNOVATION_FEATURE}, use_diou={Config.USE_DIOU_FEATURE}, "
        f"iou_feature_type={Config.IOU_FEATURE_TYPE}, "
        f"zero_mask_innov={Config.ZERO_MASK_INNOVATION}, zero_mask_diou={Config.ZERO_MASK_DIOU}"
    )
    if args.mamba_model_path:
        print('Using model:', args.mamba_model_path)
    else:
        print('Auto-detecting model based on dataset')
    set_parameters(args, args.dataset, args.mode)

    # Make result folder
    trackers_to_eval = args.pickle_path.split('/')[-1].split('.pickle')[0] + '_mamba'
    if args.tracker_suffix:
        trackers_to_eval += '_' + args.tracker_suffix
    result_folder_base = os.path.join(args.output_dir, trackers_to_eval)
    if 'dance' in args.dataset.lower() and args.mode == 'test':
        result_folder = os.path.join(result_folder_base, 'tracker')
    else:
        result_folder = result_folder_base

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder_base + '_post/', exist_ok=True)

    # Read detection result
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)
    with open(args.pickle_path_95, 'rb') as f:
        detections_95 = pickle.load(f)

    # Filter sequences if specified
    if args.sequences:
        detections = {k: v for k, v in detections.items() if k in args.sequences}
        detections_95 = {k: v for k, v in detections_95.items() if k in args.sequences}

    # Track
    total_time, total_count = track(detections, detections_95, args.data_path, result_folder, args.mode)

    # Post-processing
    if args.use_post:
        print('Running post-processing...')
        for result_file in os.listdir(result_folder):
            # Set Path
            path_in = result_folder + '/' + str(result_file)
            path_out = result_folder + '_post/' + str(result_file)
        
            # Link for DanceTrack (AFLink for non-linear dance motion)
            if 'Dance' in args.dataset:
                # Initialize AFLink only when needed
                model = PostLinker()
                model.load_state_dict(torch.load('./AFLink/AFLink_epoch20.pth'))
                aflink_dataset = LinkData('', '')
                
                linker = AFLink(path_in=path_in, path_out=path_out, model=model, dataset=aflink_dataset,
                                thrT=(0, 20), thrS=100, thrP=0.05)
                linker.link()
            
            # Linear Interpolation for SportsMOT (based on MixSort, ICCV 2023)
            # Sports motion is fast but physically constrained, short-term predictable
            elif 'Sports' in args.dataset or 'sports' in args.dataset:
                linear_interpolation_only(path_in, path_out, n_min=5, n_dti=20)
        
            # Gaussian Interpolation for MOT (pedestrian scenes)
            elif 'MOT' in args.dataset:
                gb_interpolation(path_in, path_out, interval=30, tau=12)

    # Evaluation
    if args.mode != 'test':
        print('Evaluating...')
        eval_tracker = trackers_to_eval + '_post' if args.use_post else trackers_to_eval
        if args.sequences:
            evaluate_sequences(args, eval_tracker, args.dataset, args.sequences)
        else:
            evaluate(args, eval_tracker, args.dataset)

    # Logging
    print(total_count / total_time, flush=True)
    print('', flush=True)


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Run
    run()
