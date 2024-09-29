from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
os.environ['HF_HOME'] = '/ppio_net0/huggingface'


# 创建验证集 dataloader 的函数
def get_validation_loader(dataset_name, batch_size_zs=32, num_workers=8 ):
    # 获取数据集的名称
    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 可以添加其他转换，例如归一化
    ])

    # 定义 Hugging Face 数据集名称映射
    dataset_mapping = {
        'cifar10': 'cifar10',
        'cifar100': 'cifar100',
        'caltech101': 'caltech101',
        'sun397': 'sun397',
        'flowers': 'oxford_flowers102',
        'pets': 'oxford_iiit_pet',
        'dtd': 'dtd',
        # 添加更多映射如果可用
    }

    # 检查数据集是否可用
    if dataset_name in dataset_mapping:
        hf_dataset_name = dataset_mapping[dataset_name]
        dataset = load_dataset(hf_dataset_name, split='test')
    else:
        raise ValueError(f"Unknown zero shot dataset or dataset not available in Hugging Face datasets: {dataset_name}")

    # 定义预处理函数
    def preprocess(example):
        # 对于某些数据集，图像和标签的键可能不同
        image = example.get('image', None)
        if image is None:
            image = example.get('img', None)  # 有些数据集使用 'img' 作为键

        image = transform(image)
        label = example.get('label', -1)  # 默认标签为 -1，如果不存在

        return {'image': image, 'label': label}

    # 设置数据集的转换
    dataset.set_transform(preprocess)

    # 创建验证集 dataloader
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size_zs,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    return val_loader


if __name__ == '__main__':
    a = get_validation_loader(dataset_name='caltech101')
