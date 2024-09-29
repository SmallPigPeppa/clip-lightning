from torchvision import datasets, transforms
import torch
import os

def get_statistics(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        # if images.size(1) != 3:
        #     a = images  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std


def get_cifar10(data_path):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_cifar100(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_stl10(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.STL10(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.STL10(root=data_path, split='test', download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_caltech101(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.size(0) == 1 else x),  # 这一行确保所有输入图像都是三通道的
            transforms.Normalize((0.5332, 0.5152, 0.4886), (0.2391, 0.2344, 0.2376)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.size(0) == 1 else x),  # 这一行确保所有输入图像都是三通道的
            transforms.Normalize((0.5332, 0.5152, 0.4886), (0.2391, 0.2344, 0.2376)),
        ]
    )
    dataset = datasets.Caltech101(root=data_path, target_type='category', download=True, transform=None)  # 先不进行预处理
    dataset_train, dataset_test = split_dataset(dataset)  # 划分训练集和测试集
    dataset_train.dataset.transform = transform_train  # 对训练集进行预处理
    dataset_test.dataset.transform = transform_test  # 对测试集进行预处理
    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)
    #
    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')

    return dataset_train, dataset_test


# def to_3_channels(x):
#     return torch.stack([x] * 3)


def get_fashion_mnist(data_path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.size(0) == 1 else x),  # 这一行确保所有输入图像都是三通道的,
            transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3205, 0.3205, 0.3205))
        ]
    )
    dataset_train = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
    dataset_test = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)
    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)
    #
    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')

    return dataset_train, dataset_test


def get_flowers(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5123, 0.4131, 0.3406), (0.2523, 0.2079, 0.2196)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.5123, 0.4131, 0.3406), (0.2523, 0.2079, 0.2196)),
        ]
    )
    dataset_train = datasets.Flowers102(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.Flowers102(root=data_path, split='test', download=True, transform=transform_test)

    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)
    #
    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')

    return dataset_train, dataset_test


def get_pets(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4808, 0.4424, 0.3933), (0.2161, 0.2131, 0.2138)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.4808, 0.4424, 0.3933), (0.2161, 0.2131, 0.2138)),
        ]
    )
    dataset_train = datasets.OxfordIIITPet(root=data_path, split='trainval', download=True, target_types='category',
                                           transform=transform_train)
    dataset_test = datasets.OxfordIIITPet(root=data_path, split='test', download=True, target_types='category',
                                          transform=transform_test)
    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)
    #
    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')
    return dataset_train, dataset_test

def get_cars(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4502, 0.4344, 0.4351), (0.2571, 0.2545, 0.2584)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.4502, 0.4344, 0.4351), (0.2571, 0.2545, 0.2584)),
        ]
    )
    dataset_train = datasets.StanfordCars(root=data_path, split='train', download=False, transform=transform_train)
    dataset_test = datasets.StanfordCars(root=data_path, split='test', download=False, transform=transform_test)
    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)

    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')
    return dataset_train, dataset_test


def get_aircraft(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=64, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4880, 0.5145, 0.5339), (0.1854, 0.1825, 0.2013)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4880, 0.5145, 0.5339), (0.1854, 0.1825, 0.2013)),
        ]
    )
    dataset_train = datasets.FGVCAircraft(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.FGVCAircraft(root=data_path, split='test', download=True, transform=transform_test)
    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)
    #
    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')
    return dataset_train, dataset_test


def get_rafdb(data_path):
    # train_mean = [0.57520399, 0.44951904, 0.40121641]
    # train_std = [0.20838688, 0.19108407, 0.18262798]
    # test_mean = [0.57697346, 0.44934572, 0.40011644]
    # test_std = [0.2081054, 0.18985509, 0.18132337]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.57520399, 0.44951904, 0.40121641), (0.20838688, 0.19108407, 0.18262798))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.57520399, 0.44951904, 0.40121641), (0.20838688, 0.19108407, 0.18262798))
    ])

    dataset_train = datasets.ImageFolder(os.path.join(data_path, 'RAF-DB', 'train'), transform=train_transform)
    dataset_test = datasets.ImageFolder(os.path.join(data_path, 'RAF-DB', 'test'), transform=test_transform)

    return dataset_train, dataset_test

def get_pcam(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.PCAM(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.PCAM(root=data_path, split='test', download=True, transform=transform_test)
    mean_train, std_train = get_statistics(dataset_train)
    mean_test, std_test = get_statistics(dataset_test)

    print(f'Train dataset: mean={mean_train}, std={std_train}')
    print(f'Test dataset: mean={mean_test}, std={std_test}')
    return dataset_train, dataset_test


def get_dtd(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5352, 0.4776, 0.4273), (0.1638, 0.1651, 0.1610)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.5352, 0.4776, 0.4273), (0.1638, 0.1651, 0.1610)),
        ]
    )
    dataset_train = datasets.DTD(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.DTD(root=data_path, split='test', download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_sun397(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4832, 0.4449, 0.3951), (0.2160, 0.2131, 0.2136)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            # transforms.Normalize((0.4832, 0.4449, 0.3951), (0.2160, 0.2131, 0.2136)),
        ]
    )
    dataset_train = datasets.SUN397(root=data_path, download=True, transform=transform_train)
    dataset_test = datasets.SUN397(root=data_path, download=True, transform=transform_test)
    mean_train, std_train = get_statistics(dataset_train)
    mean_test, std_test = get_statistics(dataset_test)

    print(f'Train dataset: mean={mean_train}, std={std_train}')
    print(f'Test dataset: mean={mean_test}, std={std_test}')
    return dataset_train, dataset_test


