#!/usr/bin/env python3
"""
从现有的JSON文件中提取指定序列的子集
用于自定义训练/验证划分

使用方法:
python create_inference_json.py \
    --source mot17_train.json \
    --sequences MOT17-02-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN \
    --output my_custom_split.json \
    --with-gt  # 如果需要保留GT标注（训练用）
"""

import os
import json
import argparse


def make_parser():
    parser = argparse.ArgumentParser("Extract Sequences from Existing JSON")
    
    parser.add_argument("--source", type=str, required=True,
                       help="源JSON文件名（例如: mot17_train.json）")
    parser.add_argument("--sequences", nargs='+', required=True,
                       help="要提取的序列列表，例如: MOT17-02-FRCNN MOT17-05-FRCNN")
    parser.add_argument("--output", type=str, required=True,
                       help="输出JSON文件名")
    parser.add_argument("--json_dir", type=str,
                       default="/home/shang/workspace/TrackTrack/1. YOLOX/jsons/",
                       help="JSON文件目录")
    parser.add_argument("--with-gt", action="store_true",
                       help="是否保留GT标注（默认不保留，仅推理用）")
    
    return parser


def extract_sequences(source_data, sequences, keep_gt=False):
    """从源JSON中提取指定序列"""
    
    # 找出要保留的图像
    selected_images = []
    selected_image_ids = set()
    video_id_map = {}  # 旧video_id -> 新video_id
    new_video_id = 1
    
    print(f"\n从源数据中提取序列:")
    
    for seq_name in sorted(sequences):
        # 筛选属于该序列的图像
        seq_images = [
            img for img in source_data['images'] 
            if seq_name in img['file_name']
        ]
        
        if not seq_images:
            print(f"  ✗ {seq_name} - 未找到（请检查序列名）")
            continue
        
        # 记录旧video_id到新video_id的映射
        old_video_id = seq_images[0]['video_id']
        video_id_map[old_video_id] = new_video_id
        
        # 更新video_id
        for img in seq_images:
            img['video_id'] = new_video_id
            selected_image_ids.add(img['id'])
        
        selected_images.extend(seq_images)
        print(f"  ✓ {seq_name} - {len(seq_images)} 帧")
        new_video_id += 1
    
    # 提取对应的标注（如果需要）
    selected_annotations = []
    if keep_gt and 'annotations' in source_data:
        selected_annotations = [
            ann for ann in source_data['annotations']
            if ann['image_id'] in selected_image_ids
        ]
        print(f"\n保留 {len(selected_annotations)} 个GT标注")
    else:
        print(f"\n跳过GT标注（仅推理用）")
    
    # 构建新的JSON
    result = {
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': source_data.get('categories', [{'id': 1, 'name': 'person'}]),
        'info': {
            'description': f'Custom split with sequences: {", ".join(sequences)}',
            'source': source_data.get('info', {}).get('description', 'Unknown'),
            'sequences': sequences
        }
    }
    
    return result


def main():
    args = make_parser().parse_args()
    
    # 构建源文件路径
    source_path = os.path.join(args.json_dir, args.source)
    if not os.path.exists(source_path):
        print(f"错误: 源JSON文件不存在: {source_path}")
        return 1
    
    # 读取源JSON
    print(f"读取源JSON: {args.source}")
    with open(source_path, 'r') as f:
        source_data = json.load(f)
    
    print(f"源数据统计:")
    print(f"  - 总图像数: {len(source_data['images'])}")
    print(f"  - 总标注数: {len(source_data.get('annotations', []))}")
    
    # 提取指定序列
    print(f"\n" + "="*60)
    print(f"提取序列")
    print("="*60)
    
    result_data = extract_sequences(
        source_data, 
        args.sequences,
        keep_gt=args.with_gt
    )
    
    # 保存结果
    output_path = os.path.join(args.json_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(result_data, f)
    
    print(f"\n{'='*60}")
    print(f"JSON文件生成完成!")
    print(f"{'='*60}")
    print(f"保存路径: {output_path}")
    print(f"输出统计:")
    print(f"  - 序列数: {len(args.sequences)}")
    print(f"  - 图像数: {len(result_data['images'])}")
    print(f"  - 标注数: {len(result_data['annotations'])}")
    
    # 计算文件大小
    file_size = os.path.getsize(output_path)
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024*1024):.1f} MB"
    else:
        size_str = f"{file_size / 1024:.1f} KB"
    print(f"  - 文件大小: {size_str}")
    
    print(f"\n接下来的步骤:")
    dataset = "mot17" if "MOT17" in args.sequences[0] else "mot20"
    split = "train" if "train" in args.source else "test"
    
    print(f"  1. 修改 1. YOLOX/exps/yolox_x_{dataset}_{split}.py")
    print(f"     将 self.val_ann 改为 'jsons/{args.output}'")
    print(f"\n  2. 运行检测:")
    print(f"     cd '1. YOLOX'")
    print(f"     python detect.py -f exps/yolox_x_{dataset}_{split}.py \\")
    print(f"         -c weights/{dataset}.pth.tar \\")
    print(f"         -n ../outputs/1.\\ det/{args.output.replace('.json', '.pickle')} \\")
    print(f"         --conf 0.1 --nms 0.95")
    print()
    
    return 0


if __name__ == '__main__':
    main()
