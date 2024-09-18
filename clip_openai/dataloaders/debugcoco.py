import json
import os
from PIL import Image

# 读取 captions_val2014.json 文件
json_file_path = '/Users/lwz/torch_ds/coco/annotations/captions_val2014.json'
images_folder_path = '/Users/lwz/torch_ds/coco/val2014'

# 打开并解析 JSON 文件
with open(json_file_path, 'r') as file:
    captions_data = json.load(file)

# 处理图像路径和 caption 的列表，只检查图像是否存在
image_caption_pairs = []

# 迭代 JSON 文件中的 annotations 部分，检查图像路径是否存在
for annotation in captions_data['annotations']:
    image_id = annotation['image_id']
    caption = annotation['caption']

    # 根据 image_id 生成图像文件名，确保格式为 12 位数字
    image_filename = f"COCO_val2014_{image_id:012d}.jpg"
    image_path = os.path.join(images_folder_path, image_filename)

    # 检查图像是否存在
    if os.path.exists(image_path):
        # 只记录存在的图像路径和 caption
        image_caption_pairs.append((image_path, caption))
    else:
        print(f"Image {image_path} not found.")

# 随机选择 5 个图像进行加载和显示
# 如果想按顺序显示前 5 个，可以不用随机选择
from random import sample

if len(image_caption_pairs) >= 5:
    selected_pairs = sample(image_caption_pairs, 5)  # 随机选择5个图像
else:
    selected_pairs = image_caption_pairs[:5]  # 如果少于5个，选择所有

# 加载和显示 5 个图像
for image_path, caption in selected_pairs:
    try:
        # 加载图像
        with Image.open(image_path) as image:
            # 显示 caption 和图像
            print(f"Caption: {caption}")
            image.show()  # 显示图像
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
