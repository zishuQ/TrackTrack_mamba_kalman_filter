#!/usr/bin/env python3
"""
从MOT20官方GT生成train.json (COCO格式)
排除val集使用的序列，只包含剩余的训练序列
"""

import os
import json
from collections import defaultdict

# 配置
MOT20_ROOT = "/home/shang/datasets/MOT20/train"
OUTPUT_JSON = "/home/shang/workspace/TrackTrack/1. YOLOX/jsons/mot20_train.json"

# Val集使用的序列（需要排除）
VAL_SEQS = ['MOT20-01', 'MOT20-03']

# Train集应该包含的序列
TRAIN_SEQS = ['MOT20-02', 'MOT20-05']


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
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            cls = int(parts[7])
            visibility = float(parts[8])
            
            # 只保留行人类别 (class=1) 且考虑的目标
            if cls == 1 and conf == 1:
                annotations[frame_id].append({
                    'track_id': track_id,
                    'bbox': [x, y, w, h],
                    'conf': conf,
                    'visibility': visibility
                })
    
    return annotations


def create_coco_json():
    """创建COCO格式的JSON文件"""
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'person'}]
    }
    
    image_id = 1
    annotation_id = 1
    video_id = 1
    
    print(f"处理MOT20训练序列: {TRAIN_SEQS}")
    
    for seq_name in sorted(TRAIN_SEQS):
        seq_path = os.path.join(MOT20_ROOT, seq_name)
        
        if not os.path.exists(seq_path):
            print(f"警告: 序列不存在 {seq_path}")
            continue
        
        print(f"\n处理序列: {seq_name}")
        
        # 解析序列信息
        seqinfo = parse_seqinfo(seq_path)
        seq_length = int(seqinfo['seqLength'])
        img_width = int(seqinfo['imWidth'])
        img_height = int(seqinfo['imHeight'])
        img_ext = seqinfo['imExt']
        
        # 解析GT标注
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        gt_annotations = parse_gt_txt(gt_file)
        
        # 遍历所有帧
        for frame_id in range(1, seq_length + 1):
            # 添加图像信息
            img_filename = f"MOT20/train/{seq_name}/img1/{frame_id:06d}{img_ext}"
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
                        'area': w * h
                    })
                    annotation_id += 1
            
            image_id += 1
        
        print(f"  - {seq_length} 帧处理完成")
        video_id += 1
    
    # 保存JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"\n{'='*60}")
    print(f"MOT20训练集JSON生成完成!")
    print(f"保存路径: {OUTPUT_JSON}")
    print(f"{'='*60}")
    print(f"统计信息:")
    print(f"  - 序列数: {video_id - 1}")
    print(f"  - 图像数: {len(coco_data['images'])}")
    print(f"  - 标注数: {len(coco_data['annotations'])}")
    print(f"  - 类别数: {len(coco_data['categories'])}")
    
    return coco_data


if __name__ == '__main__':
    # 检查MOT20数据集是否存在
    if not os.path.exists(MOT20_ROOT):
        print(f"错误: MOT20数据集不存在: {MOT20_ROOT}")
        print("请修改脚本中的 MOT20_ROOT 路径")
        exit(1)
    
    # 生成JSON
    create_coco_json()
    
    print("\n验证生成的文件:")
    print(f"文件大小: {os.path.getsize(OUTPUT_JSON) / 1024 / 1024:.2f} MB")
