import os
import shutil
import torch
import pickle
import argparse
import gc
import json
import numpy as np
import trackeval
import time
from pathlib import Path
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
                       choices=["off", "capture", "iwg-rg-cma"],
                       help="AgentGuard mode: off, capture, or IWG+RG-CMA")
    parser.add_argument("--agentguard-checkpoint", type=str, default=None,
                       help="Path to one combined IWG+RG-CMA checkpoint file (.pt); directories are not accepted")
    parser.add_argument(
                       "--iwg-rg-cma-output", "--rg-cma-output",
                       dest="rg_cma_output", choices=["base", "final"], default="final",
                       help="Apply base or RG-CMA-refined gate from an IWG RG-CMA checkpoint")
    parser.add_argument(
                       "--iwg-rg-cma-alpha", "--rg-cma-alpha",
                       dest="rg_cma_alpha", type=float, default=1.0,
                       help="Inference-only RG-CMA correction multiplier (alpha >= 0; 1.0 is the checkpoint final gate)")
    parser.add_argument(
                       "--agentguard-disable-kf-gate",
                       action="store_true",
                       help="Ablation: force the KF motion gate to 1.0 and keep the EMA gate learned")
    parser.add_argument(
                       "--agentguard-disable-ema-gate",
                       action="store_true",
                       help="Ablation: force the EMA appearance gate to 1.0 and keep the KF gate learned")
    parser.add_argument(
        "--agentguard-fallback-threshold",
        type=float,
        default=0.0,
        help=(
            "Inference-only conservative fallback: apply [1,1] when the "
            "largest IWG policy probability is below this threshold. "
            "0 disables the fallback."
        ),
    )
    parser.add_argument(
        "--agentguard-attention-diagnostics",
        action="store_true",
        help=(
            "Compute attention weights/entropy during IWG+RG-CMA inference. "
            "Off by default because the gates do not use these tensors."
        ),
    )
    parser.add_argument(
        "--agentguard-normalization-stats",
        type=str,
        default="",
        help="Optional norm_stats.npz overriding checkpoint scalar normalization.",
    )
    parser.add_argument(
        "--agentguard-stats-output",
        type=str,
        default="",
        help="Optional JSONL output containing final AgentGuard statistics per sequence.",
    )
    parser.add_argument("--agentguard-device", type=str, default="cpu",
                       help="Device for AgentGuard inference (cpu or cuda)")
    parser.add_argument(
        "--detection-cache-root",
        type=str,
        default="../outputs/agentguard/detection_cache",
        help="Root of per-sequence mmap detection caches. Used before legacy pickle loading.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit frames for profiling/debugging. 0 means full sequence.",
    )
    parser.add_argument(
        "--resource-log",
        type=str,
        default="",
        help="Optional JSONL path for RSS/read_bytes progress samples.",
    )
    parser.add_argument(
        "--profile-every",
        type=int,
        default=100,
        help="Write a resource sample every N frames when --resource-log is set.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Run tracking only and skip TrackEval. Useful for short profiling runs.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="PyTorch intra-op threads. AgentGuard uses many small CPU forwards; 1 keeps the desktop responsive.",
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        default=1,
        help="PyTorch inter-op threads for AgentGuard CPU inference.",
    )

    return parser


def _proc_io_read_bytes():
    try:
        with open('/proc/self/io', 'r') as f:
            for line in f:
                if line.startswith('read_bytes:'):
                    return int(line.split(':', 1)[1].strip())
    except OSError:
        return None
    return None


def _proc_rss_bytes():
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _write_resource_sample(args, sample):
    if not getattr(args, 'resource_log', ''):
        return
    path = Path(args.resource_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(sample, sort_keys=True) + '\n')


def _agentguard_stats(tracker):
    adapter = getattr(tracker, 'agentguard_adapter', None)
    runtime = getattr(adapter, 'runtime', None) if adapter is not None else None
    stats = getattr(runtime, 'stats', None) if runtime is not None else None
    if stats is None:
        return {}
    return {f"agentguard_{key}": value for key, value in stats.summary().items()}


def _write_agentguard_sequence_stats(args, tracker, sequence, elapsed_sec):
    output = getattr(args, 'agentguard_stats_output', '')
    if not output:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'sequence': sequence,
        'dataset': args.dataset,
        'split': args.mode,
        'checkpoint': getattr(args, 'agentguard_checkpoint', None),
        'normalization_stats': getattr(args, 'agentguard_normalization_stats', ''),
        'fallback_threshold': getattr(args, 'agentguard_fallback_threshold', 0.0),
        'elapsed_sec': elapsed_sec,
        **_agentguard_stats(tracker),
    }
    with path.open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')


def _detection_cache_split(dataset, mode):
    lowered = str(dataset).lower()
    if 'dance' in lowered or 'sports' in lowered:
        if mode in ('train', 'train_custom', 'all'):
            return 'train'
        if mode in ('val', 'val_custom'):
            return 'val'
        return mode
    if mode in ('val', 'val_custom', 'train', 'train_custom', 'all'):
        return 'all' if mode == 'all' else 'train'
    return mode


def _sequence_detection_cache_dir(args, vid_name):
    return (
        Path(args.detection_cache_root)
        / args.dataset
        / _detection_cache_split(args.dataset, args.mode)
        / vid_name
    )


def _has_sequence_detection_cache(args):
    if not getattr(args, 'sequences', None):
        return False
    return all((_sequence_detection_cache_dir(args, seq) / 'manifest.json').is_file() for seq in args.sequences)


def _collect_frame_results(args, data_path, track_results):
    x1y1whs, track_ids, scores = [], [], []
    for t in track_results:
        if 'MOT' in data_path and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
            continue
        if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
            x1y1whs.append(t.x1y1wh)
            track_ids.append(t.track_id)
            scores.append(t.score)
    return x1y1whs, track_ids, scores


def track_mmap_sequences(data_path, result_folder, mode):
    from agentguard.data.detection_cache import SequenceDetectionCache

    total_time, total_count = 0, 0
    for vid_name in tqdm(args.sequences, desc="Processing videos", unit="video"):
        set_parameters(args, vid_name, mode)
        with open(data_path + vid_name + '/seqinfo.ini', mode='r') as seq_info:
            for s_i in seq_info.readlines():
                if 'frameRate' in s_i:
                    args.max_time_lost = int(s_i.split('=')[-1]) * 2
                if 'imWidth' in s_i:
                    args.img_w = int(s_i.split('=')[-1])
                if 'imHeight' in s_i:
                    args.img_h = int(s_i.split('=')[-1])

        seq_cache_dir = _sequence_detection_cache_dir(args, vid_name)
        det_cache = SequenceDetectionCache(seq_cache_dir, frame_array_cache_size=4)
        tracker = Tracker(args, vid_name)
        results = []
        start_read = _proc_io_read_bytes()
        start_rss = _proc_rss_bytes()
        sequence_start = time.time()
        try:
            frame_count = det_cache.num_frames
            if args.max_frames and args.max_frames > 0:
                frame_count = min(frame_count, int(args.max_frames))
            _write_resource_sample(args, {
                'event': 'sequence_start',
                'sequence': vid_name,
                'mode': args.agentguard_mode,
                'frame': 0,
                'rss_bytes': start_rss,
                'read_bytes': start_read,
                'num_frames': frame_count,
            })
            for frame_index in tqdm(range(frame_count), desc=f"  {vid_name}", leave=False, unit="frame"):
                frame_id = frame_index + 1
                target = det_cache.get_frame(frame_index, view='target')
                source = det_cache.get_frame(frame_index, view='source')
                args.agentguard_target_detection_indices = target['detection_indices']
                args.agentguard_source_detection_indices = source['detection_indices']
                det_frame = det_cache.get_frame_array(frame_index, view='target')
                det_frame_95 = det_cache.get_frame_array(frame_index, view='source')

                start = time.time()
                if det_frame is not None and len(det_frame) > 0:
                    det_frame = inject_detection_noise(
                        det_frame,
                        noise_pos_std=args.noise_pos_std,
                        noise_size_std=args.noise_size_std,
                        drop_rate=args.drop_rate,
                    )
                    track_results = tracker.update(det_frame, det_frame_95)
                else:
                    track_results = tracker.update_without_detections()
                total_time += time.time() - start
                total_count += 1

                x1y1whs, track_ids, scores = _collect_frame_results(args, data_path, track_results)
                results.append([frame_id, track_ids, x1y1whs, scores])

                if (
                    getattr(args, 'resource_log', '')
                    and args.profile_every > 0
                    and (frame_id == 1 or frame_id % args.profile_every == 0 or frame_id == frame_count)
                ):
                    read_now = _proc_io_read_bytes()
                    rss_now = _proc_rss_bytes()
                    _write_resource_sample(args, {
                        'event': 'frame',
                        'sequence': vid_name,
                        'mode': args.agentguard_mode,
                        'frame': frame_id,
                        'elapsed_sec': time.time() - sequence_start,
                        'rss_bytes': rss_now,
                        'rss_delta_bytes': None if start_rss is None or rss_now is None else rss_now - start_rss,
                        'read_bytes': read_now,
                        'read_delta_bytes': None if start_read is None or read_now is None else read_now - start_read,
                        'active_tracks': len(getattr(tracker, 'tracks', [])),
                        'outputs': len(track_results),
                        **_agentguard_stats(tracker),
                    })

            result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
            write_results(result_filename, results)
            _write_agentguard_sequence_stats(
                args, tracker, vid_name, time.time() - sequence_start
            )
        finally:
            args.agentguard_target_detection_indices = None
            args.agentguard_source_detection_indices = None
            det_cache.close()
            del tracker
            del det_cache
            gc.collect()

    return total_time, total_count


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
            x1y1whs, track_ids, scores = _collect_frame_results(args, data_path, track_results)

            # Merge
            results.append([frame_id, track_ids, x1y1whs, scores])

        # Logging & Write results
        result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
        write_results(result_filename, results)
        _write_agentguard_sequence_stats(args, tracker, vid_name, 0.0)

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
    if args.agentguard_mode == 'capture':
        trackers_to_eval += '_agentguard_capture'
    elif args.agentguard_mode == 'iwg-rg-cma':
        trackers_to_eval += f'_iwg_rg_cma_{args.rg_cma_output}'
    result_folder_base = os.path.join(args.output_dir, trackers_to_eval)
    if 'dance' in args.dataset.lower() and args.mode == 'test':
        result_folder = os.path.join(result_folder_base, 'tracker')
        post_result_folder = os.path.join(result_folder_base + '_post', 'tracker')
    else:
        result_folder = result_folder_base
        post_result_folder = result_folder_base + '_post'

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(post_result_folder, exist_ok=True)

    if _has_sequence_detection_cache(args):
        print(f"Using per-sequence mmap detection cache from {args.detection_cache_root}")
        total_time, total_count = track_mmap_sequences(args.data_path, result_folder, args.mode)
    else:
        # Read detection result.  This legacy path may load large pickle files.
        # Single-sequence AgentGuard experiments should use the mmap detection
        # cache path above to avoid desktop stalls from repeated monolithic IO.
        detections, detections_95 = load_detection_pair(
            args.target_pickle_path,
            args.pickle_path_95,
            sequence_names=args.sequences,
        )

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
            path_in = os.path.join(result_folder, str(result_file))
            path_out = os.path.join(post_result_folder, str(result_file))
        
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
    if args.mode != "test" and not getattr(args, 'skip_eval', False):
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
    if args.agentguard_mode == 'iwg-rg-cma' and args.agentguard_checkpoint is None:
        parser.error("--agentguard-checkpoint is required when --agentguard-mode=iwg-rg-cma")

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads and args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)

    # Run
    run()
