import os
import shutil
import torch
import pickle
import argparse
import numpy as np
import trackeval
from tqdm import tqdm
from utils.etc import *
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackers.tracker import Tracker
from utils.det_feat_storage import load_detection_pair
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


def make_parser():
    parser = argparse.ArgumentParser("Tracker")

    # Basic
    parser.add_argument(
        "--pickle_dir",
        type=str,
        default="../outputs/2. det_feat/",
        help="Directory containing *_0.95.pickle and compact *_0.80.from_*_0.95.idx.pickle files.",
    )
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
    parser.add_argument("--no-reid", action="store_true", help="Disable ReID (cosine distance) in association, use IoU-only")
    parser.add_argument(
        "--kf-type",
        type=str,
        default="nsa",
        choices=["nsa", "kf", "ekf", "ukf"],
        help="Classical Kalman filter variant. 'nsa' preserves the TrackTrack baseline.",
    )

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

    # AgentGuard parameters
    parser.add_argument("--agentguard-mode", type=str, default="off",
                       choices=["off", "iwg", "full"],
                       help="AgentGuard gating mode: off (baseline), iwg (per-frame gating), full (gating + TGR replay)")
    parser.add_argument("--iwg-checkpoint", type=str, default=None,
                       help="Path to IWG model checkpoint (.pt)")
    parser.add_argument("--tgr-checkpoint", type=str, default=None,
                       help="Path to TGR model checkpoint (.pt)")
    parser.add_argument("--agentguard-device", type=str, default="cpu",
                       help="Device for AgentGuard inference (cpu or cuda)")

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

        if not hasattr(args, 'img_w'):
            raise ValueError("Image dimensions not set. Ensure seqinfo.ini is read.")

        # Set tracker
        tracker = Tracker(args, vid_name)

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

    return total_time, total_count


def run():
    # Logging & Set proper parameters
    print('Running %s %s with %s Kalman filter...' % (args.dataset, args.mode, args.kf_type))
    set_parameters(args, args.dataset, args.mode)

    # Make result folder
    tracker_base_path = getattr(args, 'target_pickle_path', args.pickle_path)
    trackers_to_eval = os.path.basename(tracker_base_path).split('.pickle')[0]
    if args.kf_type != 'nsa':
        trackers_to_eval += '_' + args.kf_type
    if hasattr(args, 'tracker_suffix') and args.tracker_suffix:
        trackers_to_eval += '_' + args.tracker_suffix
    if args.agentguard_mode == 'iwg':
        trackers_to_eval += '_agentguard_iwg'
    elif args.agentguard_mode == 'full':
        trackers_to_eval += '_agentguard_full'
    result_folder_base = os.path.join(args.output_dir, trackers_to_eval)
    if 'dance' in args.dataset.lower() and args.mode == 'test':
        result_folder = os.path.join(result_folder_base, 'tracker')
    else:
        result_folder = result_folder_base

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder_base + '_post/', exist_ok=True)

    # Read detection result
    detections, detections_95 = load_detection_pair(args.target_pickle_path, args.pickle_path_95)

    # Filter sequences if specified
    if args.sequences:
        detections = {k: v for k, v in detections.items() if k in args.sequences}
        detections_95 = {k: v for k, v in detections_95.items() if k in args.sequences}
        print(f"Filtered to sequences: {list(detections.keys())}")

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
            if 'dance' in args.dataset.lower():
                # Initialize AFLink only when needed
                model = PostLinker()
                model.load_state_dict(torch.load('./AFLink/AFLink_epoch20.pth'))
                aflink_dataset = LinkData('', '')
                
                linker = AFLink(path_in=path_in, path_out=path_out, model=model, dataset=aflink_dataset,
                                thrT=(0, 20), thrS=100, thrP=0.05)
                linker.link()
            
            # Linear Interpolation for SportsMOT (based on MixSort, ICCV 2023)
            elif 'sports' in args.dataset.lower():
                linear_interpolation_only(path_in, path_out, n_min=5, n_dti=20)
        
            # Gaussian Interpolation for MOT (pedestrian scenes)
            elif 'mot' in args.dataset.lower():
                gb_interpolation(path_in, path_out, interval=30, tau=12)
            else:
                # If no filter hits, copy plain results to post folder to avoid missing files in evaluation.
                shutil.copy(path_in, path_out)

    # Evaluation
    if args.mode != "test":
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
    parser = make_parser()
    args = parser.parse_args()

    # AgentGuard validation
    if args.agentguard_mode == 'iwg' and args.iwg_checkpoint is None:
        parser.error("--iwg-checkpoint is required when --agentguard-mode=iwg")
    if args.agentguard_mode == 'full':
        if args.iwg_checkpoint is None:
            parser.error("--iwg-checkpoint is required when --agentguard-mode=full")
        if args.tgr_checkpoint is None:
            parser.error("--tgr-checkpoint is required when --agentguard-mode=full")

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Run
    run()
