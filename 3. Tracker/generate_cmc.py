#!/usr/bin/env python3
"""
生成完整的GMC（Global Motion Compensation）文件
基于BoT-SORT的实现: https://github.com/NirAharon/BoT-SORT
使用Sparse Optical Flow (Lucas-Kanade) 估计帧间的相机运动
"""
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser("Generate GMC files (BoT-SORT method)")
    parser.add_argument("--dataset", type=str, default="MOT17", choices=["MOT17", "MOT20"])
    parser.add_argument("--data_dir", type=str, default="/home/shang/datasets/")
    parser.add_argument("--sequences", nargs='+', help="指定序列，例如: MOT17-02-FRCNN MOT17-13-FRCNN。不指定则处理所有序列")
    parser.add_argument("--output_dir", type=str, default="./trackers/cmc/")
    parser.add_argument("--downscale", type=int, default=2, help="图像下采样倍数 (加速计算)")
    return parser


def estimate_camera_motion_sparse_optflow(prev_gray, curr_gray, prev_keypoints=None, downscale=2):
    """
    使用Sparse Optical Flow (Lucas-Kanade)估计相机运动
    这是BoT-SORT使用的方法
    
    Args:
        prev_gray: 前一帧灰度图
        curr_gray: 当前帧灰度图  
        prev_keypoints: 前一帧的关键点 (如果None则重新检测)
        downscale: 下采样倍数
        
    Returns:
        H: 2x3 相似变换矩阵
        keypoints: 当前帧检测的关键点 (用于下一帧)
    """
    H = np.eye(2, 3, dtype=np.float64)
    
    # 下采样加速
    height, width = prev_gray.shape
    if downscale > 1:
        prev_gray = cv2.resize(prev_gray, (width // downscale, height // downscale))
        curr_gray = cv2.resize(curr_gray, (width // downscale, height // downscale))
    
    # 检测关键点 (Shi-Tomasi角点检测)
    if prev_keypoints is None:
        feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, 
                             blockSize=3, useHarrisDetector=False, k=0.04)
        prev_keypoints = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        
        if prev_keypoints is None:
            return H, None
    
    # 使用Lucas-Kanade光流跟踪关键点
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    curr_keypoints, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, 
                                                             prev_keypoints, None, **lk_params)
    
    if curr_keypoints is None:
        # 检测新的关键点用于下一帧
        feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, 
                             blockSize=3, useHarrisDetector=False, k=0.04)
        new_keypoints = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
        return H, new_keypoints
    
    # 过滤出成功跟踪的点
    prev_points = []
    curr_points = []
    for i in range(len(status)):
        if status[i]:
            prev_points.append(prev_keypoints[i])
            curr_points.append(curr_keypoints[i])
    
    if len(prev_points) < 4:
        # 点太少，检测新的关键点
        feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, 
                             blockSize=3, useHarrisDetector=False, k=0.04)
        new_keypoints = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
        return H, new_keypoints
    
    prev_points = np.array(prev_points)
    curr_points = np.array(curr_points)
    
    # 估计相似变换 (Similarity Transform)
    try:
        H, inliers = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
        
        if H is None:
            H = np.eye(2, 3, dtype=np.float64)
        else:
            # 恢复原始尺度的平移
            if downscale > 1:
                H[0, 2] *= downscale
                H[1, 2] *= downscale
    except:
        H = np.eye(2, 3, dtype=np.float64)
    
    # 检测新的关键点用于下一帧
    feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, 
                         blockSize=3, useHarrisDetector=False, k=0.04)
    new_keypoints = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
    
    return H, new_keypoints


def generate_gmc_for_sequence(seq_path, seq_name, output_path, downscale=2):
    """为单个序列生成GMC文件 (使用Sparse Optical Flow方法)"""
    
    # 读取序列信息
    seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
    seqinfo = {}
    with open(seqinfo_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('['):
                key, value = line.strip().split('=', 1)
                seqinfo[key] = value
    
    seq_length = int(seqinfo['seqLength'])
    img_ext = seqinfo['imExt']
    img_dir = os.path.join(seq_path, 'img1')
    
    print(f"\n处理序列: {seq_name}")
    print(f"  帧数: {seq_length}")
    print(f"  图像路径: {img_dir}")
    
    # 打开输出文件
    with open(output_path, 'w') as f:
        # 第一帧使用单位矩阵
        f.write("0\t1.000000\t0.000000\t0.000000\t0.000000\t1.000000\t0.000000\t\n")
        
        # 读取第一帧
        prev_img_path = os.path.join(img_dir, f"000001{img_ext}")
        prev_img = cv2.imread(prev_img_path)
        if prev_img is None:
            print(f"错误: 无法读取第一帧 {prev_img_path}")
            return False
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        prev_keypoints = None  # 第一帧没有关键点
        
        # 逐帧处理
        for frame_id in tqdm(range(2, seq_length + 1), desc=f"  生成GMC"):
            # 读取当前帧
            curr_img_path = os.path.join(img_dir, f"{frame_id:06d}{img_ext}")
            curr_img = cv2.imread(curr_img_path)
            
            if curr_img is None:
                # 图像读取失败，使用单位矩阵
                warp_matrix = np.eye(2, 3, dtype=np.float64)
                curr_keypoints = None
            else:
                curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
                # 估计相机运动
                warp_matrix, curr_keypoints = estimate_camera_motion_sparse_optflow(
                    prev_gray, curr_gray, prev_keypoints, downscale)
                prev_gray = curr_gray
                prev_keypoints = curr_keypoints
            
            # 写入文件 (格式: frame_id \t 6个仿射矩阵参数 \t)
            f.write(f"{frame_id-1}\t{warp_matrix[0,0]:.6f}\t{warp_matrix[0,1]:.6f}\t"
                   f"{warp_matrix[0,2]:.6f}\t{warp_matrix[1,0]:.6f}\t"
                   f"{warp_matrix[1,1]:.6f}\t{warp_matrix[1,2]:.6f}\t\n")
    
    print(f"  ✓ 完成: {output_path}")
    return True


def verify_cmc_file(generated_file, reference_file=None):
    """验证生成的CMC文件格式和数值"""
    data = []
    with open(generated_file, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) >= 7:
                data.append([float(x) for x in tokens[:7]])
    data = np.array(data)
    
    print(f"\n  验证: {os.path.basename(generated_file)}")
    print(f"    总帧数: {len(data)}")
    print(f"    相似变换约束 (a11=a22): {np.allclose(data[:,1], data[:,5], atol=1e-6)}")
    print(f"    相似变换约束 (a12=-a21): {np.allclose(data[:,2], -data[:,4], atol=1e-6)}")
    print(f"    scale 范围: [{data[:,1].min():.6f}, {data[:,1].max():.6f}], mean={data[:,1].mean():.6f}")
    print(f"    rotation 范围: [{data[:,2].min():.6f}, {data[:,2].max():.6f}], mean={data[:,2].mean():.6f}")
    
    if reference_file and os.path.exists(reference_file):
        ref_data = []
        with open(reference_file, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) >= 7:
                    ref_data.append([float(x) for x in tokens[:7]])
        ref_data = np.array(ref_data)
        
        if len(ref_data) == len(data):
            diff = np.abs(data - ref_data)
            print(f"\n    对比参考文件: {os.path.basename(reference_file)}")
            print(f"      平均差异: {diff[:, 1:].mean():.6f}")
            print(f"      最大差异: {diff[:, 1:].max():.6f}")
            if diff[:, 1:].max() < 1e-4:
                print(f"      ✓ 数值几乎一致!")
            else:
                print(f"      ⚠ 数值有差异（这是正常的，不同算法/参数会产生不同结果）")


def main():
    args = make_parser().parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要处理的序列
    dataset_path = os.path.join(args.data_dir, args.dataset, 'train')
    
    if args.sequences:
        sequences = args.sequences
    else:
        # 获取所有序列
        sequences = [d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))]
        sequences = sorted(sequences)
    
    print(f"{'='*60}")
    print(f"生成GMC文件 (BoT-SORT Sparse Optical Flow方法)")
    print(f"{'='*60}")
    print(f"数据集: {args.dataset}")
    print(f"下采样: {args.downscale}x")
    print(f"序列数: {len(sequences)}")
    
    # 处理每个序列
    success_count = 0
    for seq_name in sequences:
        seq_path = os.path.join(dataset_path, seq_name)
        if not os.path.isdir(seq_path):
            print(f"\n跳过: {seq_name} (不是目录)")
            continue
        
        # 生成输出文件名
        if 'MOT17' in seq_name:
            output_name = f"GMC-{seq_name.split('-FRCNN')[0]}.txt"
        elif 'MOT20' in seq_name:
            output_name = f"GMC-{seq_name}.txt"
        else:
            output_name = f"GMC-{seq_name}.txt"
        
        output_path = os.path.join(args.output_dir, output_name)
        
        # 生成GMC
        if generate_gmc_for_sequence(seq_path, seq_name, output_path, args.downscale):
            success_count += 1
            
            # 验证生成的文件
            verify_cmc_file(output_path)
    
    print(f"\n{'='*60}")
    print(f"完成! 成功生成 {success_count}/{len(sequences)} 个GMC文件")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
