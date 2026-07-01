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
from trackers.alt_kalman_filter_wrapper import AltKalmanFilterWrapper
from utils.det_feat_storage import load_detection_pair
from utils.gbi import gb_interpolation, linear_interpolation_only


def make_parser():
    parser = argparse.ArgumentParser("Tracker with AltKalmanFilter")

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

    # For trackers
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--max_time_lost", type=float, default=30)
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)

    # Alt Kalman Filter
    parser.add_argument(
        "--alt_model_path",
        type=str,
        default=None,
        help="Path to AltKalmanFilter model weights.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="lstm",
        choices=["lstm", "gru", "rnn", "transformer", "mlp", "tcn", "s4d"],
        help="Alternative backbone type.",
    )
    parser.add_argument("--tracker-suffix", type=str, default="",
                       help="Suffix appended to tracker output folder name for unique results")
    parser.add_argument("--sequences", type=str, nargs="+", default=None,
                       help="Only track specific sequences (e.g. --sequences MOT20-01)")
    return parser


def track(detections, detections_95, data_path, result_folder, mode):
    total_time, total_count = 0, 0
    vid_names = list(detections.keys())
    for vid_name in tqdm(vid_names, desc="Processing videos", unit="video"):
        set_parameters(args, vid_name, mode)

        with open(data_path + vid_name + '/seqinfo.ini', mode='r') as seq_info:
            for s_i in seq_info.readlines():
                if 'frameRate' in s_i:
                    args.max_time_lost = int(s_i.split('=')[-1]) * 2
                if 'imWidth' in s_i:
                    args.img_w = int(s_i.split('=')[-1])
                if 'imHeight' in s_i:
                    args.img_h = int(s_i.split('=')[-1])

        tracker = TrackerMamba(args, vid_name)
        tracker.shared_kalman_filter = AltKalmanFilterWrapper.get_shared_instance(
            model_path=args.alt_model_path,
            device='cuda',
            backbone=args.backbone,
        )
        tracker.shared_kalman_filter.set_image_size(args.img_w, args.img_h)

        results = []
        frame_ids = sorted(detections[vid_name].keys())
        for frame_id in tqdm(frame_ids, desc=f"  {vid_name}", leave=False, unit="frame"):
            start = time.time()
            if detections[vid_name][frame_id] is not None:
                track_results = tracker.update(detections[vid_name][frame_id], detections_95[vid_name][frame_id])
            else:
                track_results = tracker.update_without_detections()
            total_time += time.time() - start
            total_count += 1

            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                if 'MOT' in data_path and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                    continue
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)
            results.append([frame_id, track_ids, x1y1whs, scores])

        result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
        write_results(result_filename, results)

        for trk in tracker.tracks:
            tracker.shared_kalman_filter.delete_track(trk.track_id)

    return total_time, total_count


def run():
    print('Running %s %s with AltKalmanFilter (%s)...' % (args.dataset, args.mode, args.backbone))
    if args.alt_model_path:
        print('Using model:', args.alt_model_path)
    else:
        print('Warning: no --alt_model_path provided, using random weights.')
    set_parameters(args, args.dataset, args.mode)

    tracker_base_path = getattr(args, 'target_pickle_path', args.pickle_path)
    trackers_to_eval = os.path.basename(tracker_base_path).split('.pickle')[0] + f'_alt_{args.backbone}'
    if hasattr(args, 'tracker_suffix') and args.tracker_suffix:
        trackers_to_eval += '_' + args.tracker_suffix
    result_folder_base = os.path.join(args.output_dir, trackers_to_eval)
    if 'dance' in args.dataset.lower() and args.mode == 'test':
        result_folder = os.path.join(result_folder_base, 'tracker')
    else:
        result_folder = result_folder_base

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder_base + '_post/', exist_ok=True)

    detections, detections_95 = load_detection_pair(args.target_pickle_path, args.pickle_path_95)

    # Filter sequences if specified
    if args.sequences:
        detections = {k: v for k, v in detections.items() if k in args.sequences}
        detections_95 = {k: v for k, v in detections_95.items() if k in args.sequences}
        print(f"Filtered to sequences: {list(detections.keys())}")

    total_time, total_count = track(detections, detections_95, args.data_path, result_folder, args.mode)

    if args.use_post:
        print('Running post-processing...')
        for result_file in os.listdir(result_folder):
            path_in = result_folder + '/' + str(result_file)
            path_out = result_folder + '_post/' + str(result_file)

            if 'Dance' in args.dataset:
                model = PostLinker()
                model.load_state_dict(torch.load('./AFLink/AFLink_epoch20.pth'))
                aflink_dataset = LinkData('', '')
                linker = AFLink(
                    path_in=path_in,
                    path_out=path_out,
                    model=model,
                    dataset=aflink_dataset,
                    thrT=(0, 20),
                    thrS=100,
                    thrP=0.05,
                )
                linker.link()
            elif 'Sports' in args.dataset or 'sports' in args.dataset:
                linear_interpolation_only(path_in, path_out, n_min=5, n_dti=20)
            elif 'MOT' in args.dataset:
                gb_interpolation(path_in, path_out, interval=30, tau=12)

    if args.mode != 'test':
        print('Evaluating...')
        eval_tracker = trackers_to_eval + '_post' if args.use_post else trackers_to_eval
        if args.sequences:
            import numpy as np
            import trackeval
            import tempfile
            from utils.etc import get_trackeval_configs
            eval_config, dataset_config = get_trackeval_configs(args, eval_tracker, args.dataset)
            seqmap_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, prefix='seqmap_')
            seqmap_tmp.write('name\n')
            for seq in args.sequences:
                seqmap_tmp.write(seq + '\n')
            seqmap_tmp.close()
            dataset_config['SEQMAP_FILE'] = seqmap_tmp.name
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
            res, _ = evaluator.evaluate(dataset_list, metrics_list)
            hota = np.mean(res['MotChallenge2DBox'][eval_tracker]['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']).item()
            idf1 = res['MotChallenge2DBox'][eval_tracker]['COMBINED_SEQ']['pedestrian']['Identity']['IDF1']
            mota = res['MotChallenge2DBox'][eval_tracker]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA']
            assa = np.mean(res['MotChallenge2DBox'][eval_tracker]['COMBINED_SEQ']['pedestrian']['HOTA']['AssA']).item()
            deta = np.mean(res['MotChallenge2DBox'][eval_tracker]['COMBINED_SEQ']['pedestrian']['HOTA']['DetA']).item()
            print(f'{"HOTA":<10}{"MOTA":<10}{"IDF1":<10}{"DetA":<10}{"AssA":<10}', flush=True)
            print(f'{hota:<10.6f}{mota:<10.6f}{idf1:<10.6f}{deta:<10.6f}{assa:<10.6f}', flush=True)
            os.unlink(seqmap_tmp.name)
        else:
            evaluate(args, eval_tracker, args.dataset)

    print(total_count / total_time, flush=True)
    print('', flush=True)


if __name__ == "__main__":
    args = make_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    run()
