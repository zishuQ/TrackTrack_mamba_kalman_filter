import os
import glob
import re
from functools import lru_cache
from flask import Flask, jsonify, request, send_file, render_template, abort

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.abspath(
    os.path.join(APP_ROOT, "..", "..", "outputs", "3. track")
)

DATASET_ROOTS = {
    "DanceTrack": os.path.expanduser("~/datasets/DanceTrack"),
    "MOT17": os.path.expanduser("~/datasets/MOT17"),
    "MOT20": os.path.expanduser("~/datasets/MOT20"),
    "SportsMOT": os.path.expanduser("~/datasets/SportsMOT/dataset"),
}

ENV_ROOT_OVERRIDES = {
    "DanceTrack": "TRACKVIEW_DANCETRACK_ROOT",
    "MOT17": "TRACKVIEW_MOT17_ROOT",
    "MOT20": "TRACKVIEW_MOT20_ROOT",
    "SportsMOT": "TRACKVIEW_SPORTSMOT_ROOT",
}

GT_DATASETS = {"MOT17", "MOT20", "SportsMOT"}

for name, env_key in ENV_ROOT_OVERRIDES.items():
    if os.environ.get(env_key):
        DATASET_ROOTS[name] = os.path.expanduser(os.environ[env_key])

app = Flask(__name__, template_folder="templates", static_folder="static")


def _safe_join(base, *paths):
    base = os.path.abspath(base)
    candidate = os.path.abspath(os.path.join(base, *paths))
    if os.path.commonpath([candidate, base]) != base:
        raise ValueError("Unsafe path")
    return candidate


def _detect_dataset(folder_name):
    lower = folder_name.lower()
    if lower.startswith("dance"):
        return "DanceTrack"
    if lower.startswith("mot17"):
        return "MOT17"
    if lower.startswith("mot20"):
        return "MOT20"
    if lower.startswith("sportsmot"):
        return "SportsMOT"
    return None


def _safe_candidate_join(base, *paths):
    try:
        return _safe_join(base, *paths)
    except ValueError:
        return None


def _detect_mode(folder_name):
    lower = folder_name.lower()
    patterns = (
        ("val_custom", r"(?:^|[_-])val_custom(?:[_-]|$)"),
        ("train_custom", r"(?:^|[_-])train_custom(?:[_-]|$)"),
        ("all", r"(?:^|[_-])all(?:[_-]|$)"),
        ("val", r"(?:^|[_-])val(?:[_-]|$)"),
        ("test", r"(?:^|[_-])test(?:[_-]|$)"),
        ("train", r"(?:^|[_-])train(?:[_-]|$)"),
    )
    for mode, pattern in patterns:
        if re.search(pattern, lower):
            return mode
    return None


def _physical_split_candidates(dataset, mode):
    if dataset in {"MOT17", "MOT20"}:
        if mode == "test":
            return ["test"]
        if mode in {"train", "val", "val_custom", "train_custom", "all"}:
            return ["train"]
        return ["train", "test"]

    if dataset == "SportsMOT":
        if mode in {"train", "val", "test"}:
            return [mode]
        return ["val", "train", "test"]

    if dataset == "DanceTrack":
        if mode in {"train", "val", "test"}:
            return [mode]
        return ["val", "test", "train"]

    return [mode] if mode else []


def _sequence_root_candidates(dataset, mode, sequence):
    root = DATASET_ROOTS.get(dataset)
    if not root:
        return []

    split_candidates = _physical_split_candidates(dataset, mode)
    candidate_parts = []

    if dataset == "DanceTrack":
        for split in split_candidates:
            candidate_parts.extend(
                [
                    ("tracker", split, sequence),
                    (split, sequence),
                ]
            )
        candidate_parts.extend(
            [
                ("tracker", sequence),
                (sequence,),
            ]
        )
    else:
        for split in split_candidates:
            candidate_parts.append((split, sequence))
        candidate_parts.append((sequence,))

    candidates = []
    for parts in candidate_parts:
        candidate = _safe_candidate_join(root, *parts)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _sequence_dir(dataset, mode, sequence):
    candidates = [os.path.join(root, "img1") for root in _sequence_root_candidates(dataset, mode, sequence)]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    # 如果没有找到目录，则仍然返回最常见路径，供后续错误处理使用
    return candidates[0] if candidates else None


def _gt_file(dataset, mode, sequence):
    if dataset not in GT_DATASETS:
        return None

    for root in _sequence_root_candidates(dataset, mode, sequence):
        candidate = os.path.join(root, "gt", "gt.txt")
        if os.path.isfile(candidate):
            return candidate
    return None


@lru_cache(maxsize=256)
def _resolve_sequence_context(folder, sequence):
    dataset = _detect_dataset(folder)
    mode = _detect_mode(folder)
    img_dir = _sequence_dir(dataset, mode, sequence) if dataset else None
    gt_file = _gt_file(dataset, mode, sequence) if dataset else None
    return {
        "dataset": dataset,
        "mode": mode,
        "img_dir": img_dir,
        "gt_file": gt_file,
        "has_gt": bool(gt_file and os.path.isfile(gt_file)),
    }


def _find_image_path(img_dir, frame):
    if not img_dir or not os.path.isdir(img_dir):
        return None
    candidates = [
        f"{frame:06d}.jpg",
        f"{frame:06d}.png",
        f"{frame:08d}.jpg",
        f"{frame:08d}.png",
        f"{frame:04d}.jpg",
        f"{frame:04d}.png",
    ]
    for name in candidates:
        path = os.path.join(img_dir, name)
        if os.path.exists(path):
            return path
    patterns = [
        os.path.join(img_dir, f"*{frame:06d}*.jpg"),
        os.path.join(img_dir, f"*{frame:06d}*.png"),
        os.path.join(img_dir, f"*{frame:08d}*.jpg"),
        os.path.join(img_dir, f"*{frame:08d}*.png"),
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


@lru_cache(maxsize=32)
def _load_mot_file(file_path):
    frame_map = {}
    min_frame = None
    max_frame = None

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            parsed = _parse_box_parts(parts)
            if not parsed:
                continue
            frame, track_id, x, y, w, h = parsed
            score = float(parts[6]) if len(parts) > 6 else 1.0

            if min_frame is None or frame < min_frame:
                min_frame = frame
            if max_frame is None or frame > max_frame:
                max_frame = frame

            frame_map.setdefault(frame, []).append(
                {
                    "id": track_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "score": score,
                }
            )

    if min_frame is None:
        min_frame = 1
        max_frame = 1

    return frame_map, min_frame, max_frame


def _parse_box_parts(parts):
    if len(parts) < 6:
        return None

    try:
        frame = int(float(parts[0]))
        object_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
    except ValueError:
        return None

    return frame, object_id, x, y, w, h


def _keep_gt_record(dataset, parts):
    if dataset in {"MOT17", "MOT20"}:
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        class_id = int(float(parts[7])) if len(parts) > 7 else 1
        return conf == 1.0 and class_id == 1

    if dataset == "SportsMOT":
        visibility = float(parts[8]) if len(parts) > 8 else 1.0
        if visibility < 0.25:
            return False

        ignore_flag = int(float(parts[6])) if len(parts) > 6 else 1
        if ignore_flag != 1:
            return False

        class_id = int(float(parts[7])) if len(parts) > 7 else 1
        if class_id in {3, 4, 5, 6, 9, 10, 11}:
            return False
        if class_id in {2, 7, 8, 12}:
            return False
        return True

    return False


@lru_cache(maxsize=64)
def _load_gt_file(file_path, dataset):
    if not file_path or not os.path.isfile(file_path):
        return {}

    frame_map = {}
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            parsed = _parse_box_parts(parts)
            if not parsed or not _keep_gt_record(dataset, parts):
                continue
            frame, gt_id, x, y, w, h = parsed
            frame_map.setdefault(frame, []).append(
                {
                    "id": gt_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )
    return frame_map


def _iou(left, right):
    left_x2 = left["x"] + left["w"]
    left_y2 = left["y"] + left["h"]
    right_x2 = right["x"] + right["w"]
    right_y2 = right["y"] + right["h"]

    inter_x1 = max(left["x"], right["x"])
    inter_y1 = max(left["y"], right["y"])
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    left_area = left["w"] * left["h"]
    right_area = right["w"] * right["h"]
    return inter_area / max(left_area + right_area - inter_area, 1e-6)


def _match_gt_tracker(gt_boxes, tracker_boxes, iou_threshold=0.3):
    pairs = []
    for gt_box in gt_boxes:
        for tracker_box in tracker_boxes:
            score = _iou(gt_box, tracker_box)
            if score > iou_threshold:
                pairs.append((score, gt_box["id"], tracker_box["id"]))

    pairs.sort(reverse=True)
    tracker_to_gt = {}
    used_gt = set()
    used_tracker = set()

    for _, gt_id, tracker_id in pairs:
        if gt_id in used_gt or tracker_id in used_tracker:
            continue
        tracker_to_gt[tracker_id] = gt_id
        used_gt.add(gt_id)
        used_tracker.add(tracker_id)

    return tracker_to_gt


def _annotate_tracker_frames(tracker_frames, gt_frames):
    annotated_frames = {}
    for frame, boxes in tracker_frames.items():
        tracker_to_gt = _match_gt_tracker(gt_frames.get(frame, []), boxes)
        annotated_frames[frame] = []
        for box in boxes:
            annotated_box = dict(box)
            gt_id = tracker_to_gt.get(box["id"])
            if gt_id is not None:
                annotated_box["gt_id"] = gt_id
            annotated_frames[frame].append(annotated_box)
    return annotated_frames


@lru_cache(maxsize=64)
def _load_sequence_payload(file_path, dataset, gt_file):
    tracker_frames, min_frame, max_frame = _load_mot_file(file_path)
    has_gt = bool(dataset in GT_DATASETS and gt_file and os.path.isfile(gt_file))
    if not has_gt:
        return tracker_frames, min_frame, max_frame, False

    gt_frames = _load_gt_file(gt_file, dataset)
    return _annotate_tracker_frames(tracker_frames, gt_frames), min_frame, max_frame, True


def _resolve_result_file(folder, sequence):
    folder_path = _safe_join(RESULT_ROOT, folder)
    candidates = [
        os.path.join(folder_path, f"{sequence}.txt"),
        os.path.join(folder_path, "tracker", f"{sequence}.txt"),
    ]
    for file_path in candidates:
        if os.path.exists(file_path):
            return file_path
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/folders")
def list_folders():
    folders = []
    if os.path.isdir(RESULT_ROOT):
        for name in sorted(os.listdir(RESULT_ROOT)):
            full = os.path.join(RESULT_ROOT, name)
            if os.path.isdir(full):
                folders.append(name)
    return jsonify({"folders": folders})


@app.route("/api/sequences")
def list_sequences():
    folder = request.args.get("folder", "")
    if not folder:
        return jsonify({"sequences": []})
    folder_path = _safe_join(RESULT_ROOT, folder)
    sequences = []
    if os.path.isdir(folder_path):
        for name in sorted(os.listdir(folder_path)):
            if name.endswith(".txt"):
                sequences.append(name[:-4])
    tracker_folder = os.path.join(folder_path, "tracker")
    if not sequences and os.path.isdir(tracker_folder):
        for name in sorted(os.listdir(tracker_folder)):
            if name.endswith(".txt"):
                sequences.append(name[:-4])
    return jsonify({"sequences": sequences})


@app.route("/api/frames")
def frame_range():
    folder = request.args.get("folder", "")
    sequence = request.args.get("sequence", "")
    if not folder or not sequence:
        return jsonify({"min": 1, "max": 1, "has_gt": False})
    file_path = _resolve_result_file(folder, sequence)
    if not file_path:
        return jsonify({"min": 1, "max": 1, "has_gt": False})
    context = _resolve_sequence_context(folder, sequence)
    _, min_frame, max_frame = _load_mot_file(file_path)
    return jsonify({"min": min_frame, "max": max_frame, "has_gt": context["has_gt"]})


@app.route("/api/sequence")
def sequence_data():
    folder = request.args.get("folder", "")
    sequence = request.args.get("sequence", "")
    if not folder or not sequence:
        return jsonify({"min": 1, "max": 1, "frames": {}, "has_gt": False})
    file_path = _resolve_result_file(folder, sequence)
    if not file_path:
        return jsonify({"min": 1, "max": 1, "frames": {}, "has_gt": False})

    context = _resolve_sequence_context(folder, sequence)
    frame_map, min_frame, max_frame, has_gt = _load_sequence_payload(
        file_path, context["dataset"], context["gt_file"]
    )
    frames = {str(frame): boxes for frame, boxes in frame_map.items()}
    return jsonify({"min": min_frame, "max": max_frame, "frames": frames, "has_gt": has_gt})


@app.route("/api/frame")
def frame_data():
    folder = request.args.get("folder", "")
    sequence = request.args.get("sequence", "")
    frame_str = request.args.get("frame", "1")
    try:
        frame = int(frame_str)
    except ValueError:
        frame = 1

    file_path = _resolve_result_file(folder, sequence)
    if not file_path:
        return jsonify({"frame": frame, "boxes": [], "image": None, "has_gt": False})

    context = _resolve_sequence_context(folder, sequence)
    frame_map, min_frame, max_frame, has_gt = _load_sequence_payload(
        file_path, context["dataset"], context["gt_file"]
    )
    boxes = frame_map.get(frame, [])

    img_path = _find_image_path(context["img_dir"], frame) if context["img_dir"] else None

    return jsonify(
        {
            "frame": frame,
            "min": min_frame,
            "max": max_frame,
            "boxes": boxes,
            "image": "",
            "has_image": bool(img_path),
            "has_gt": has_gt,
        }
    )


@app.route("/api/image")
def frame_image():
    folder = request.args.get("folder", "")
    sequence = request.args.get("sequence", "")
    frame_str = request.args.get("frame", "1")
    try:
        frame = int(frame_str)
    except ValueError:
        frame = 1

    context = _resolve_sequence_context(folder, sequence)
    img_path = _find_image_path(context["img_dir"], frame) if context["img_dir"] else None
    if not img_path:
        abort(404)
    return send_file(img_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
