import os
import glob
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

for name, env_key in ENV_ROOT_OVERRIDES.items():
    if os.environ.get(env_key):
        DATASET_ROOTS[name] = os.path.expanduser(os.environ[env_key])

app = Flask(__name__, template_folder="templates", static_folder="static")


def _safe_join(base, *paths):
    candidate = os.path.abspath(os.path.join(base, *paths))
    if not candidate.startswith(base):
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


def _detect_split(folder_name):
    lower = folder_name.lower()
    if "_val" in lower:
        return "val"
    if "_test" in lower or lower.endswith("test"):
        return "test"
    if "_train" in lower or lower.endswith("train"):
        return "train"
    return "test"


def _sequence_dir(dataset, split, sequence):
    root = DATASET_ROOTS.get(dataset)
    if not root:
        return None

    candidates = []
    # DanceTrack 数据集在不同版本中会放在根目录下或 tracker 子目录中
    if dataset == "DanceTrack":
        candidates = [
            os.path.join(root, "tracker", split, sequence, "img1"),
            os.path.join(root, "tracker", sequence, "img1"),
            os.path.join(root, split, sequence, "img1"),
            os.path.join(root, sequence, "img1"),
        ]
    else:
        candidates = [
            os.path.join(root, split, sequence, "img1"),
            os.path.join(root, sequence, "img1"),
        ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    # 如果没有找到目录，则仍然返回最常见路径，供后续错误处理使用
    return candidates[0] if candidates else None


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
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
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
        return jsonify({"min": 1, "max": 1})
    file_path = _resolve_result_file(folder, sequence)
    if not file_path:
        return jsonify({"min": 1, "max": 1})
    _, min_frame, max_frame = _load_mot_file(file_path)
    return jsonify({"min": min_frame, "max": max_frame})


@app.route("/api/sequence")
def sequence_data():
    folder = request.args.get("folder", "")
    sequence = request.args.get("sequence", "")
    if not folder or not sequence:
        return jsonify({"min": 1, "max": 1, "frames": {}})
    file_path = _resolve_result_file(folder, sequence)
    if not file_path:
        return jsonify({"min": 1, "max": 1, "frames": {}})

    frame_map, min_frame, max_frame = _load_mot_file(file_path)
    frames = {str(frame): boxes for frame, boxes in frame_map.items()}
    return jsonify({"min": min_frame, "max": max_frame, "frames": frames})


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
        return jsonify({"frame": frame, "boxes": [], "image": None})

    frame_map, min_frame, max_frame = _load_mot_file(file_path)
    boxes = frame_map.get(frame, [])

    dataset = _detect_dataset(folder)
    split = _detect_split(folder)
    img_dir = _sequence_dir(dataset, split, sequence) if dataset else None
    img_path = _find_image_path(img_dir, frame) if img_dir else None

    return jsonify(
        {
            "frame": frame,
            "min": min_frame,
            "max": max_frame,
            "boxes": boxes,
            "image": "",
            "has_image": bool(img_path),
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

    dataset = _detect_dataset(folder)
    split = _detect_split(folder)
    img_dir = _sequence_dir(dataset, split, sequence) if dataset else None
    img_path = _find_image_path(img_dir, frame) if img_dir else None
    if not img_path:
        abort(404)
    return send_file(img_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
