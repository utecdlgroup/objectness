import os

import torchvision
import torchvision.transforms as transforms

available_datasets = ["mnist", "celeba", "cifar10", "svhn"]

def load_dataset(dataroot, dataset_name, image_size, num_channels, **kwargs):
    if dataset_name not in available_datasets:
        raise Exception("Unsupported dataset '{}'. Available datasets: '{}''.".format(
            dataset_name, "', '".join(available_datasets)
        ))


    if dataset_name == 'mnist':
        dataset_function = torchvision.datasets.MNIST
    elif dataset_name == 'celeba':
        dataset_function = torchvision.datasets.CelebA
    elif dataset_name == 'cifar10':
        dataset_function = torchvision.datasets.CIFAR10
    elif dataset_name == 'svhn':
        dataset_function = torchvision.datasets.SVHN



    if num_channels not in [1, 3]:
        raise Exception("Given number of channels ({}) is unsupported. Only 1 or 3 channels can be used.".format(
            num_channels
        ))


    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    if num_channels == 1 and dataset_name != 'mnist':
        transform_list.append(transforms.Grayscale())
    elif num_channels == 3 and dataset_name == 'mnist':
        raise Exception('Error: Conversion of MNIST to 3 channels is not supported.')

    transform_list.append(transforms.Normalize((0.5,), (0.5,)))


    transform = transforms.Compose(transform_list)

    dataset = dataset_function(root=os.path.join(dataroot, dataset_name), transform=transform, **kwargs)


    return dataset