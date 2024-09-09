import os
import pandas as pd
from PIL import Image
import numpy as np
from .base import ImageRetrievalDataset
from concurrent.futures import ThreadPoolExecutor


class UPMCFood101Dataset(ImageRetrievalDataset):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def fetch_dataset(self, split):
        # 如果 split 是 'val'，将其转换为 'test'
        if split == 'val':
            split = 'test'

        # 确定图像和CSV文件的路径
        self.root_dir = os.path.join(self.root_dir, 'UPMC-Food101')
        images_dir = os.path.join(self.root_dir, 'images', split)
        csv_file = os.path.join(self.root_dir, 'texts', f'{split}_titles.csv')

        # 加载CSV文件，使用逗号作为分隔符
        data = pd.read_csv(csv_file, delimiter=',', header=None, names=['image', 'title', 'category'])

        # 确保图像和类别是字符串
        data['image'] = data['image'].astype(str)
        data['category'] = data['category'].astype(str)

        # 获取图像路径和标题
        selected_images_paths = [
            os.path.join(images_dir, category, img)
            for img, category in zip(data['image'], data['category'])
        ]

        images_titles = data['title'].tolist()

        return selected_images_paths, images_titles

    def get_resolution_statistics(self, split, batch_size=128):
        from concurrent.futures import ThreadPoolExecutor

        selected_images_paths, _ = self.fetch_dataset(split)

        all_widths = []
        all_heights = []
        pixel_sum = np.zeros(3)
        pixel_squared_sum = np.zeros(3)
        total_pixels = 0

        def _process_image(path):
            with Image.open(path) as img:
                width, height = img.size
                img_array = np.array(img).astype(np.float32) / 255.0  # 归一化到 [0, 1]

                if img_array.ndim == 2:  # 灰度图像，转换为伪RGB
                    img_array = np.stack([img_array] * 3, axis=-1)

                # 更新分辨率信息
                return width, height, img_array

        def process_batch(batch):
            nonlocal pixel_sum, pixel_squared_sum, total_pixels
            widths = []
            heights = []
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(_process_image, batch))

            for width, height, img_array in results:
                widths.append(width)
                heights.append(height)
                total_pixels += img_array.shape[0] * img_array.shape[1]
                pixel_sum += img_array.sum(axis=(0, 1))
                pixel_squared_sum += (img_array ** 2).sum(axis=(0, 1))

            return widths, heights

        for i in range(0, len(selected_images_paths), batch_size):
            batch = selected_images_paths[i:i + batch_size]
            widths, heights = process_batch(batch)
            all_widths.extend(widths)
            all_heights.extend(heights)

        # 计算每个通道的 mean 和 std
        mean_pixel_value = pixel_sum / total_pixels
        std_pixel_value = np.sqrt(pixel_squared_sum / total_pixels - mean_pixel_value ** 2)

        resolution_stats = {
            "max_width": np.max(all_widths),
            "min_width": np.min(all_widths),
            "mean_width": np.mean(all_widths),
            "max_height": np.max(all_heights),
            "min_height": np.min(all_heights),
            "mean_height": np.mean(all_heights),
        }

        # 自动打印结果
        print(f"{split.capitalize()} split - Mean Pixel Value: {mean_pixel_value}")
        print(f"{split.capitalize()} split - Std Pixel Value: {std_pixel_value}")
        print(f"{split.capitalize()} split - Resolution Stats: {resolution_stats}")

        return mean_pixel_value, std_pixel_value, resolution_stats
