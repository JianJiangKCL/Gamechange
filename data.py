from torchvision import datasets
import torchvision.transforms as transforms


def create_dataset(dataset_name, dataset_path, train=True):
    dataset = dataset_name.lower()
    assert dataset in ['mnist', 'cifar10', 'omniglot']
    if dataset == 'mnist':
        ds = datasets.MNIST(dataset_path, train=train, transform=create_transform(dataset, train), download=True)
        num_classes, in_channels = 10, 1
    elif dataset == 'cifar10':
        ds = datasets.CIFAR10(dataset_path, train=train, transform=create_transform(dataset, train), download=False)

        num_classes, in_channels = 10, 3

    elif dataset == 'omniglot':
        ds = datasets.Omniglot(dataset_path, background=train, transform=create_transform(dataset, train), download=False)
        num_classes, in_channels = 1623, 1

    return ds, num_classes, in_channels


def create_transform(dataset, train=True):
    assert dataset in ['mnist', 'cifar10']

    def get_normalization(dataset):
        if dataset == 'mnist':
            mean, std = (0.1307,), (0.3081,)
        elif dataset == 'cifar10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        return transforms.Normalize(mean, std)

    transform = []
    if train:
        if dataset == 'cifar10':
            transform.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(p=0.5), ])
        # elif dataset == 'mnist':
        #     transform.extend([transforms.Resize(32)])
    transform.extend([transforms.ToTensor(), get_normalization(dataset)])
    return transforms.Compose(transform)


