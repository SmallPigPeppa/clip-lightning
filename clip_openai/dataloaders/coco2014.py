import os
import json
from .base import ImageRetrievalDataset


class COCO2014Dataset(ImageRetrievalDataset):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    # def fetch_dataset(self, split):
    #     self.root_dir = os.path.join(self.root_dir, 'coco')
    #     json_path = os.path.join(self.root_dir, 'annotations', 'dataset.json')
    #     with open(json_path, 'r') as file:
    #         all_data = json.load(file)['images']
    #
    #     if split == 'train':
    #         split_data = [item for item in all_data if item['split'] in ['train', 'restval']]
    #     elif split == 'val':
    #         split_data = [item for item in all_data if item['split'] == 'test']
    #     else:
    #         raise ValueError('Split must be either "train" or "val"')
    #
    #     images = []
    #     captions = []
    #
    #     for item in split_data:
    #         img = os.path.join(self.root_dir, item['filepath'], item['filename'])
    #         caps = [sentence['raw'] for sentence in item['sentences']]
    #         assert os.path.isfile(img)
    #         images.append(img)
    #         captions.append(caps)
    #
    #     return images, captions

    def fetch_dataset(self, split):
        self.root_dir = os.path.join(self.root_dir, 'coco')

        # 根据不同的 split 选择对应的 JSON 文件
        if split == 'train':
            json_path = os.path.join(self.root_dir, 'annotations', 'captions_train2014.json')
        elif split == 'val':
            json_path = os.path.join(self.root_dir, 'annotations', 'captions_val2014.json')
        else:
            raise ValueError("Split must be either 'train' or 'val'")

        # 打开并读取 JSON 文件
        with open(json_path, 'r') as file:
            all_data = json.load(file)

        # 字典用于存储图像路径和对应的多个 caption
        image_caption_dict = {}

        # 迭代所有的 annotations，处理图像路径和 captions
        for annotation in all_data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']

            # 根据 image_id 生成图像文件名，确保格式为 12 位数字
            image_filename = f"COCO_{split}2014_{image_id:012d}.jpg"
            image_path = os.path.join(self.root_dir, f'{split}2014', image_filename)

            # 检查图像路径是否存在
            if os.path.exists(image_path):
                if image_path not in image_caption_dict:
                    # 如果图像路径尚不存在字典中，初始化一个空列表
                    image_caption_dict[image_path] = []

                # 将 caption 添加到图像对应的列表中
                image_caption_dict[image_path].append(caption)
            else:
                # 路径不存在时抛出 FileNotFoundError
                raise FileNotFoundError(f"Image {image_path} not found.")

        # 返回图像路径和对应的 caption 列表
        return image_caption_dict
