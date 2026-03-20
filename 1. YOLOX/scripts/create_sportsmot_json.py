#!/usr/bin/env python3
"""
从SportsMOT官方GT生成JSON (COCO格式)
基于官方 mot_to_coco.py 修改
https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/convert_mot_to_coco.py
"""
import os
import numpy as np
import json
import cv2
from tqdm import tqdm

# 配置
DATA_PATH = "/home/shang/datasets/SportsMOT/dataset"
OUTPUT_DIR = "/home/shang/workspace/TrackTrack/1. YOLOX/jsons"
SPLITS = ["train", "val", "test"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split)
        if not os.path.exists(data_path):
            print(f"跳过 {split}: 目录不存在 {data_path}")
            continue
        
        out_path = os.path.join(OUTPUT_DIR, f"sportsmot_{split}.json")
        out = {
            "images": [],
            "annotations": [],
            "videos": [],
            "categories": [{"id": 1, "name": "pedestrian"}]
        }
        
        video_list = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        
        print(f"\n{'='*60}")
        print(f"生成 SportsMOT {split} JSON...")
        print(f"{'='*60}")
        
        for seq in tqdm(sorted(video_list), desc=f"Processing {split}"):
            if ".DS_Store" in seq:
                continue
            if not os.path.isdir(os.path.join(data_path, seq)):
                continue
                
            video_cnt += 1
            out["videos"].append({"id": video_cnt, "file_name": seq})
            
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, "img1")
            ann_path = os.path.join(seq_path, "gt/gt.txt")
            
            images = os.listdir(img_path)
            num_images = len([image for image in images if "jpg" in image.lower()])
            
            image_range = [0, num_images - 1]
            
            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                    
                img_file = os.path.join(data_path, f"{seq}/img1/{i + 1:06d}.jpg")
                img = cv2.imread(img_file)
                if img is None:
                    print(f"警告: 无法读取图像 {img_file}")
                    continue
                height, width = img.shape[:2]
                
                image_info = {
                    "file_name": f"SportsMOT/dataset/{split}/{seq}/img1/{i + 1:06d}.jpg",
                    "id": image_cnt + i + 1,
                    "frame_id": i + 1 - image_range[0],
                    "prev_image_id": image_cnt + i if i > 0 else -1,
                    "next_image_id": image_cnt + i + 2 if i < num_images - 1 else -1,
                    "video_id": video_cnt,
                    "height": height,
                    "width": width
                }
                out["images"].append(image_info)
            
            print(f"  {seq}: {num_images} images")
            
            # 处理标注 (test集可能没有GT)
            if os.path.exists(ann_path):
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
                if len(anns.shape) == 1:
                    anns = anns.reshape(1, -1)
                
                print(f"  {seq}: {int(anns[:, 0].max())} ann frames")
                
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    
                    track_id = int(anns[i][1])
                    
                    # 过滤条件
                    if anns.shape[1] > 8:
                        visibility = float(anns[i][8])
                        if visibility < 0.25:
                            continue
                    
                    if anns.shape[1] > 6:
                        ignore_flag = int(anns[i][6])
                        if ignore_flag != 1:
                            continue
                    
                    if anns.shape[1] > 7:
                        class_id = int(anns[i][7])
                        # 非人类类别跳过
                        if class_id in [3, 4, 5, 6, 9, 10, 11]:
                            continue
                        # 被忽略的人类
                        if class_id in [2, 7, 8, 12]:
                            category_id = -1
                        else:
                            category_id = 1
                    else:
                        category_id = 1
                    
                    ann_cnt += 1
                    ann = {
                        "id": ann_cnt,
                        "category_id": category_id,
                        "image_id": image_cnt + frame_id,
                        "track_id": track_id,
                        "bbox": anns[i][2:6].tolist(),
                        "conf": float(anns[i][6]) if anns.shape[1] > 6 else 1.0,
                        "iscrowd": 0,
                        "area": float(anns[i][4] * anns[i][5])
                    }
                    out["annotations"].append(ann)
            
            image_cnt += num_images
        
        print(f"\n统计信息:")
        print(f"  - 视频数: {len(out['videos'])}")
        print(f"  - 图像数: {len(out['images'])}")
        print(f"  - 标注数: {len(out['annotations'])}")
        print(f"  - 保存路径: {out_path}")
        
        with open(out_path, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
