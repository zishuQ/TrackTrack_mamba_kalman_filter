#!/usr/bin/env python3
"""
从DanceTrack官方GT生成JSON (COCO格式)
支持train/val/test集
"""

import os
import json
from collections import defaultdict

# 配置
DANCETRACK_ROOT = "/home/shang/datasets/DanceTrack"
OUTPUT_DIR = "/home/shang/workspace/TrackTrack/1. YOLOX/jsons"


def parse_seqinfo(seq_path):
    """解析seqinfo.ini获取序列信息"""
    seqinfo_file = os.path.join(seq_path, 'seqinfo.ini')
    info = {}
    with open(seqinfo_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('['):
                key, value = line.split('=', 1)
                info[key] = value
    return info


def parse_gt_txt(gt_file):
    """解析gt.txt文件
    格式: frame_id, track_id, x, y, w, h, conf, class, visibility
    """
    annotations = defaultdict(list)
    if not os.path.exists(gt_file):
        return annotations
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(parts[7]) if len(parts) > 7 else 1
            visibility = float(parts[8]) if len(parts) > 8 else 1.0
            
            # 只保留行人类别
            if cls == 1:
                annotations[frame_id].append({
                    'track_id': track_id,
                    'bbox': [x, y, w, h],
                    'conf': conf,
                    'visibility': visibility
                })
    
    return annotations


def create_coco_json(split, data_dir, has_gt=True):
    """创建COCO格式的JSON文件"""
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'person'}]
    }
    
    image_id = 1
    annotation_id = 1
    video_id = 1
    
    # 获取序列列表
    seq_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"处理{split}集序列: {len(seq_names)} 个")
    
    for seq_name in seq_names:
        seq_path = os.path.join(data_dir, seq_name)
        
        if not os.path.exists(seq_path):
            print(f"警告: 序列不存在 {seq_path}")
            continue
        
        print(f"  处理序列: {seq_name}")
        
        # 解析序列信息
        seqinfo = parse_seqinfo(seq_path)
        seq_length = int(seqinfo['seqLength'])
        img_width = int(seqinfo['imWidth'])
        img_height = int(seqinfo['imHeight'])
        img_ext = seqinfo.get('imExt', '.jpg')
        
        # 解析GT标注（如果有）
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        gt_annotations = parse_gt_txt(gt_file) if has_gt else {}
        
        # 遍历所有帧
        for frame_id in range(1, seq_length + 1):
            # 添加图像信息 - DanceTrack使用8位数字格式
            img_filename = f"DanceTrack/{split}/{seq_name}/img1/{frame_id:08d}{img_ext}"
            coco_data['images'].append({
                'file_name': img_filename,
                'id': image_id,
                'video_id': video_id,
                'frame_id': frame_id,
                'height': img_height,
                'width': img_width
            })
            
            # 添加该帧的标注
            if frame_id in gt_annotations:
                for ann in gt_annotations[frame_id]:
                    x, y, w, h = ann['bbox']
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'category_id': 1,
                        'image_id': image_id,
                        'track_id': ann['track_id'],
                        'bbox': [x, y, w, h],
                        'conf': ann['conf'],
                        'iscrowd': 0,
                        'area': w * h,
                        'visibility': ann['visibility']
                    })
                    annotation_id += 1
            
            image_id += 1
        
        video_id += 1
    
    return coco_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 定义数据集split
    splits = {
        'val': (os.path.join(DANCETRACK_ROOT, 'val'), True),
        'test': (os.path.join(DANCETRACK_ROOT, 'test1'), False),  # test没有GT
        'train': (os.path.join(DANCETRACK_ROOT, 'train1'), True),
    }
    
    for split_name, (data_dir, has_gt) in splits.items():
        if not os.path.exists(data_dir):
            print(f"跳过 {split_name}: 目录不存在 {data_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"生成 DanceTrack {split_name} JSON...")
        print(f"{'='*60}")
        
        coco_data = create_coco_json(split_name, data_dir, has_gt)
        
        # 保存JSON
        output_file = os.path.join(OUTPUT_DIR, f"dancetrack_{split_name}.json")
        with open(output_file, 'w') as f:
            json.dump(coco_data, f)
        
        print(f"\n统计信息:")
        print(f"  - 图像数: {len(coco_data['images'])}")
        print(f"  - 标注数: {len(coco_data['annotations'])}")
        print(f"  - 保存路径: {output_file}")


if __name__ == '__main__':
    main()
