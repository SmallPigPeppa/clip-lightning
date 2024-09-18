import json
import os
from PIL import Image

# 读取 captions_val2014.json 文件
json_file_path = '/Users/lwz/torch_ds/coco/annotations/captions_val2014.json'
images_folder_path = '/Users/lwz/torch_ds/coco/val2014'

# 打开并解析 JSON 文件
with open(json_file_path, 'r') as file:
    captions_data = json.load(file)

# 处理图像和 caption 的列表
image_caption_pairs = []

# 迭代 JSON 文件中的 annotations 部分，加载图像和 caption
for annotation in captions_data['annotations']:
    image_id = annotation['image_id']
    caption = annotation['caption']

    # 根据 image_id 生成图像文件名，确保格式为 12 位数字
    image_filename = f"COCO_val2014_{image_id:012d}.jpg"
    image_path = os.path.join(images_folder_path, image_filename)

    # 检查图像是否存在
    if os.path.exists(image_path):
        try:
            # 加载图像
            image = Image.open(image_path)
            # 将图像和 caption 加入列表
            image_caption_pairs.append((image, caption))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    else:
        print(f"Image {image_path} not found.")

# 现在 image_caption_pairs 包含所有图像和对应的 caption
# 示例：显示前几个图像和 caption
for i in range(5):
    image, caption = image_caption_pairs[i]
    print(f"Caption: {caption}")
    image.show()  # 显示图像