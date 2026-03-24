import os
import sys
import torch
import pickle
import argparse
from tqdm import tqdm

# Add parent directory to path to access mamba_kalman_filter module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.etc import *
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackers.tracker_mamba import TrackerMamba
from trackers.mamba_kalman_filter_wrapper import MambaKalmanFilterWrapper
from utils.gbi import gb_interpolation, linear_interpolation_only


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
                track_results = tracker.update(detections[vid_name][frame_id], detections_95[vid_name][frame_id])
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
    # Logging & Set proper parameters
    print('Running %s %s with MambaKalmanFilter...' % (args.dataset, args.mode))
    if args.mamba_model_path:
        print('Using model:', args.mamba_model_path)
    else:
        print('Auto-detecting model based on dataset')
    set_parameters(args, args.dataset, args.mode)

    # Make result folder
    trackers_to_eval = args.pickle_path.split('/')[-1].split('.pickle')[0] + '_mamba'
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
