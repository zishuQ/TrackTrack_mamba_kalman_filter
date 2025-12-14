import json

with open('/home/shang/workspace/TrackTrack/1. YOLOX/jsons/mot20_train.json', 'r') as f:
    train_data = json.load(f)

with open('/home/shang/workspace/TrackTrack/1. YOLOX/jsons/mot20_train.json', 'r') as f:
    val_data = json.load(f)

print(f"Train: {len(train_data['images'])} 图像, {len(train_data['annotations'])} 标注")
print(f"Val: {len(val_data['images'])} 图像, {len(val_data['annotations'])} 标注")

# 构建train中的文件名集合（用于快速查找）
train_filenames = set(img['file_name'] for img in train_data['images'])

print(f"\nTrain中的唯一文件数: {len(train_filenames)}")
print("\n" + "="*60)
print("检查 Val 中的所有图像是否都在 Train 中:")
print("="*60)

# 检查val中的每个文件是否在train中
val_in_train = 0
val_not_in_train = []

for val_img in val_data['images']:
    if val_img['file_name'] in train_filenames:
        val_in_train += 1
    else:
        val_not_in_train.append(val_img['file_name'])

print(f"\nVal中在Train里的图像: {val_in_train} / {len(val_data['images'])}")
print(f"Val中不在Train里的图像: {len(val_not_in_train)} / {len(val_data['images'])}")

if len(val_not_in_train) == 0:
    print("\n✅ 结论: 所有Val数据都包含在Train中")
else:
    print("\n❌ 结论: Val中有图像不在Train中")
    print(f"\n不在Train中的图像（前10个）:")
    for fname in val_not_in_train[:10]:
        print(f"  - {fname}")

# 显示一些重复的例子
print("\n" + "="*60)
print("Val和Train重复的图像示例（前5个）:")
print("="*60)
count = 0
for val_img in val_data['images']:
    if val_img['file_name'] in train_filenames:
        print(f"  {val_img['file_name']}")
        count += 1
        if count >= 5:
            break