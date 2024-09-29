from torchvision import datasets
data_path='/ppio_net0/torch_ds'
# dataset_train = datasets.SUN397(root=data_path, download=True, transform=None)
dataset_test = datasets.STL10(root=data_path, split='test', download=True, transform=None)
dataset_test = datasets.OxfordIIITPet(root=data_path, split='test', download=True, target_types='category',transform=None)
dataset_test = datasets.StanfordCars(root=data_path, split='test', download=False, transform=None)
dataset_test = datasets.FGVCAircraft(root=data_path, split='test', download=True, transform=None)
