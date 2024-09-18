import os
import json
from tqdm import tqdm  # 导入 tqdm 以显示进度条


def generate_new_captions_json(root_dir, split):
    # 根据不同的 split 选择对应的 JSON 文件
    if split == 'train':
        json_path = os.path.join(root_dir, 'annotations', 'captions_train2014.json')
    elif split == 'val':
        json_path = os.path.join(root_dir, 'annotations', 'captions_val2014.json')
    else:
        raise ValueError("Split must be either 'train' or 'val'")

    # 新的 JSON 文件路径
    new_json_path = os.path.join(root_dir, 'annotations', f'captions_{split}2014_new.json')

    # 读取原始 JSON 文件
    with open(json_path, 'r') as file:
        all_data = json.load(file)

    # 字典用于存储图像路径和对应的 caption 列表
    new_data = {}

    # 获取 annotations 列表的长度
    annotations = all_data['annotations']
    total_annotations = len(annotations)

    # 迭代所有的 annotations，生成新的 image_path 和对应的 caption 列表
    for annotation in tqdm(annotations, desc=f"Processing {split} captions", total=total_annotations):
        image_id = annotation['image_id']
        caption = annotation['caption']

        # 根据 image_id 生成相对图像文件路径，确保格式为 12 位数字
        image_filename = f"COCO_{split}2014_{image_id:012d}.jpg"
        image_path = os.path.join(f'{split}2014', image_filename)  # 相对路径

        # 如果图像路径已经存在，则将 caption 添加到对应的列表中
        if image_path not in new_data:
            new_data[image_path] = []

        new_data[image_path].append(caption)

    # 统计图片和标题的数量
    num_images = len(new_data)
    num_captions = sum(len(captions) for captions in new_data.values())

    # 将结果保存到新的 JSON 文件中
    with open(new_json_path, 'w') as new_file:
        json.dump(new_data, new_file, indent=4)

    print(f"\nNew captions JSON saved to: {new_json_path}")
    print(f"Total images in {split}: {num_images}")
    print(f"Total captions in {split}: {num_captions}\n")


if __name__ == "__main__":
    # COCO 数据集的根目录
    root_dir = '/Users/lwz/torch_ds/coco'  # 替换为你的 COCO 数据集路径

    # 生成新的 captions_train2014_new.json 文件
    generate_new_captions_json(root_dir, 'train')

    # 生成新的 captions_val2014_new.json 文件
    generate_new_captions_json(root_dir, 'val')
