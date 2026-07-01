import argparse
import os
import pickle
import random
import shutil
import time

import numpy as np
import torch
import trackeval
from tqdm import tqdm

from AFLink.AppFreeLink import AFLink
from AFLink.dataset import LinkData
from AFLink.model import PostLinker
from trackers.apu_diff_adapter import APUDiffAdapter
from trackers.tracker_apu import TrackerAPU
from utils.det_feat_storage import load_detection_pair
from utils.etc import evaluate, evaluate_sequences, set_parameters, write_results
from utils.gbi import gb_interpolation, linear_interpolation_only


def inject_detection_noise(detections_frame, noise_pos_std=0.0, noise_size_std=0.0, drop_rate=0.0):
    if detections_frame is None or len(detections_frame) == 0:
        return detections_frame
    det = detections_frame.copy()
    if noise_pos_std > 0:
        n = len(det)
        dx = np.random.normal(0, noise_pos_std, size=n)
        dy = np.random.normal(0, noise_pos_std, size=n)
        det[:, 0] += dx
        det[:, 2] += dx
        det[:, 1] += dy
        det[:, 3] += dy
    if noise_size_std > 0:
        n = len(det)
        dw = np.random.normal(0, noise_size_std, size=n)
        dh = np.random.normal(0, noise_size_std, size=n)
        det[:, 0] -= dw / 2
        det[:, 2] += dw / 2
        det[:, 1] -= dh / 2
        det[:, 3] += dh / 2
    if drop_rate > 0 and len(det) > 0:
        keep_mask = np.random.random(len(det)) >= drop_rate
        if keep_mask.sum() == 0:
            keep_mask[np.random.randint(len(det))] = True
        det = det[keep_mask]
    return det


def make_parser():
    parser = argparse.ArgumentParser("Tracker with APUDiff appearance adapter")
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
    parser.add_argument("--use_post", action="store_true")
    parser.add_argument("--print-per-sequence-metrics", action="store_true")

    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--max_time_lost", type=float, default=30)
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--kf-type", type=str, default="nsa", choices=["nsa", "kf", "ekf", "ukf"])

    parser.add_argument("--apu-model-path", type=str, default="/home/shang/workspace/diffusion_appearance/checkpoints/apu_diff_full.pth")
    parser.add_argument("--apu-repo-dir", type=str, default="/home/shang/workspace/diffusion_appearance")
    parser.add_argument("--apu-device", type=str, default="cuda")
    parser.add_argument("--apu-sample-steps", type=int, default=1, help="APUDiff sampling steps. Default keeps one-step DiffMOT-style inference.")
    parser.add_argument("--apu-stochastic", action="store_true", help="Use stochastic APUDiff sampling for ablation. Default is deterministic.")
    parser.add_argument(
        "--apu-history-update-mode",
        type=str,
        default="observed",
        choices=["observed", "predicted"],
        help=(
            "APUDiff appearance history update after a successful match. "
            "'observed' appends the matched detection feature; 'predicted' appends APUDiff prediction for closed-loop ablation."
        ),
    )
    

    parser.add_argument("--noise-pos-std", type=float, default=0.0)
    parser.add_argument("--noise-size-std", type=float, default=0.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--disable-gmc", action="store_true")
    parser.add_argument("--tracker-suffix", type=str, default="")
    parser.add_argument("--sequences", type=str, nargs="+", default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="Debug only: stop each sequence after this many frames.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip TrackEval, useful for partial max-frame smoke runs.")
    return parser


def track(args, detections, detections_95, data_path, result_folder, mode):
    total_time, total_count = 0, 0
    apu_adapter = APUDiffAdapter.get_shared_instance(
        model_path=args.apu_model_path,
        repo_dir=args.apu_repo_dir,
        device=args.apu_device,
        sample_steps=args.apu_sample_steps,
        stochastic=args.apu_stochastic,
        history_update_mode=args.apu_history_update_mode,
    )
    for vid_name in tqdm(list(detections.keys()), desc="Processing videos", unit="video"):
        set_parameters(args, vid_name, mode)
        with open(data_path + vid_name + "/seqinfo.ini", mode="r") as seq_info:
            for s_i in seq_info.readlines():
                if "frameRate" in s_i:
                    args.max_time_lost = int(s_i.split("=")[-1]) * 2
                if "imWidth" in s_i:
                    args.img_w = int(s_i.split("=")[-1])
                if "imHeight" in s_i:
                    args.img_h = int(s_i.split("=")[-1])

        tracker = TrackerAPU(args, vid_name, apu_adapter)
        results = []
        frame_ids = sorted(detections[vid_name].keys())
        if getattr(args, "max_frames", 0) > 0:
            frame_ids = frame_ids[: args.max_frames]
        for frame_id in tqdm(frame_ids, desc=f"  {vid_name}", leave=False, unit="frame"):
            start = time.time()
            if detections[vid_name][frame_id] is not None:
                det_frame = inject_detection_noise(
                    detections[vid_name][frame_id],
                    noise_pos_std=args.noise_pos_std,
                    noise_size_std=args.noise_size_std,
                    drop_rate=args.drop_rate,
                )
                track_results = tracker.update(det_frame, detections_95[vid_name][frame_id])
            else:
                track_results = tracker.update_without_detections()
            total_time += time.time() - start
            total_count += 1

            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                if "MOT" in data_path and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                    continue
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)
            results.append([frame_id, track_ids, x1y1whs, scores])
        write_results(os.path.join(result_folder, f"{vid_name}.txt"), results)
    return total_time, total_count


def run(args):
    print(f"Running {args.dataset} {args.mode} with APUDiff appearance adapter...")
    print("Using APUDiff model:", args.apu_model_path)
    set_parameters(args, args.dataset, args.mode)

    tracker_base_path = getattr(args, "target_pickle_path", args.pickle_path)
    trackers_to_eval = os.path.basename(tracker_base_path).split(".pickle")[0] + "_apu_diff"
    if args.kf_type != "nsa":
        trackers_to_eval += "_" + args.kf_type
    if args.tracker_suffix:
        trackers_to_eval += "_" + args.tracker_suffix
    result_folder_base = os.path.join(args.output_dir, trackers_to_eval)
    result_folder = os.path.join(result_folder_base, "tracker") if "dance" in args.dataset.lower() and args.mode == "test" else result_folder_base
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder_base + "_post/", exist_ok=True)

    detections, detections_95 = load_detection_pair(args.target_pickle_path, args.pickle_path_95)
    if args.sequences:
        detections = {k: v for k, v in detections.items() if k in args.sequences}
        detections_95 = {k: v for k, v in detections_95.items() if k in args.sequences}
        print(f"Filtered to sequences: {list(detections.keys())}")

    total_time, total_count = track(args, detections, detections_95, args.data_path, result_folder, args.mode)
    APUDiffAdapter.get_shared_instance(
        model_path=args.apu_model_path,
        repo_dir=args.apu_repo_dir,
        device=args.apu_device,
        sample_steps=args.apu_sample_steps,
        stochastic=args.apu_stochastic,
        history_update_mode=args.apu_history_update_mode,
    ).report_stats()

    if args.use_post:
        print("Running post-processing...")
        for result_file in os.listdir(result_folder):
            path_in = result_folder + "/" + str(result_file)
            path_out = result_folder + "_post/" + str(result_file)
            if "dance" in args.dataset.lower():
                model = PostLinker()
                model.load_state_dict(torch.load("./AFLink/AFLink_epoch20.pth"))
                linker = AFLink(path_in=path_in, path_out=path_out, model=model, dataset=LinkData("", ""), thrT=(0, 20), thrS=100, thrP=0.05)
                linker.link()
            elif "sports" in args.dataset.lower():
                linear_interpolation_only(path_in, path_out, n_min=5, n_dti=20)
            elif "mot" in args.dataset.lower():
                gb_interpolation(path_in, path_out, interval=30, tau=12)
            else:
                shutil.copy(path_in, path_out)

    if args.mode != "test" and not args.skip_eval and getattr(args, "max_frames", 0) <= 0:
        print("Evaluating...")
        eval_tracker = trackers_to_eval + "_post" if args.use_post else trackers_to_eval
        if args.sequences:
            evaluate_sequences(args, eval_tracker, args.dataset, args.sequences)
        else:
            evaluate(args, eval_tracker, args.dataset)
    elif args.mode != "test":
        print("Skipping evaluation for partial/debug run.")
    print(total_count / total_time, flush=True)
    print("", flush=True)


if __name__ == "__main__":
    args = make_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    run(args)
