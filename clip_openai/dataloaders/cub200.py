import os
import json
from .base import ImageRetrievalDataset


class CUB200Dataset(ImageRetrievalDataset):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def fetch_dataset(self, split):
        # 加载图像路径和ID
        images_path = {}
        self.root_dir = os.path.join(self.root_dir, 'cub200')
        with open(os.path.join(self.root_dir, 'CUB_200_2011', 'images.txt')) as f:
            for line in f:
                image_id, path = line.strip().split()
                images_path[image_id] = path

        # 确定是训练集还是验证集的图像ID
        images_id = []
        with open(os.path.join(self.root_dir, 'CUB_200_2011', 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_train = line.strip().split()
                if (split == 'train' and int(is_train) == 1) or (split == 'val' and int(is_train) == 0):
                    images_id.append(image_id)

        # 获取所有选中图像的路径
        selected_images_paths = [os.path.join(self.root_dir, 'CUB_200_2011', 'images', images_path[id]) for id in
                                 images_id if id in images_path]
        images_captions = []

        # 对每一个图像，加载对应的所有caption
        for image_id in images_id:
            if image_id in images_path:
                captions_path = os.path.join(self.root_dir, 'cvpr2016_cub', 'text_c10',
                                             images_path[image_id].replace('.jpg', '.txt'))
                with open(captions_path) as f:
                    captions = [line.strip() for line in f.readlines()]
                images_captions.append(captions)

        return selected_images_paths, images_captions
