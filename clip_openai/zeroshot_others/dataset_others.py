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
        transforms.Resize((224, 224)),
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
        # [
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        # ]
        [
            transforms.Resize((336, 336)),
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.STL10(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.STL10(root=data_path, split='test', download=True, transform=transform_test)
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4880, 0.5145, 0.5339), (0.1854, 0.1825, 0.2013)),
        ]
    )
    dataset_train = datasets.FGVCAircraft(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.FGVCAircraft(root=data_path, split='test', download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_rafdb(data_path):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.57520399, 0.44951904, 0.40121641), (0.20838688, 0.19108407, 0.18262798))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
            transforms.Resize((224, 224)),
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

def get_food101(data_path):
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
            transforms.Normalize((0.5545, 0.4432, 0.3324), (0.2222, 0.2340, 0.2303)),
        ]
    )
    dataset_train = datasets.Food101(root=data_path, download=True, transform=transform_train)
    dataset_test = datasets.Food101(root=data_path, download=True, transform=transform_test)
    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)

    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')
    return dataset_train, dataset_test


def get_dataset(data_path, dataset_name):
    dataset_name = dataset_name.lower()  # 转换为小写，以避免大小写差异导致的错误

    if dataset_name == 'cifar10':
        _, dataset_val = get_cifar10(data_path)
    elif dataset_name == 'cifar100':
        _, dataset_val = get_cifar100(data_path)
    elif dataset_name == 'stl10':
        _, dataset_val = get_stl10(data_path)
    elif dataset_name == 'flowers':
        _, dataset_val = get_flowers(data_path)
    elif dataset_name == 'pets':
        _, dataset_val = get_pets(data_path)
    elif dataset_name == 'cars':
        _, dataset_val = get_cars(data_path)
    elif dataset_name == 'aircraft':
        _, dataset_val = get_aircraft(data_path)
    elif dataset_name == 'rafdb':
        _, dataset_val = get_rafdb(data_path)
    elif dataset_name == 'pcam':
        _, dataset_val = get_pcam(data_path)
    elif dataset_name == 'dtd':
        _, dataset_val = get_dtd(data_path)
    elif dataset_name == 'sun397':
        _, dataset_val = get_sun397(data_path)
    elif dataset_name == 'food101':
        _, dataset_val = get_food101(data_path)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return dataset_val
