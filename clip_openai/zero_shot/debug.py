from torchvision import datasets, transforms
data_path='/ppio_net0/torch_ds'
datasets.Caltech101(root=data_path, target_type='category', download=True, transform=None)