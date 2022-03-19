import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor


def get_data(batch_size):
    one_hot_encoding = Lambda(
        lambda label: torch.zeros(10, dtype=torch.float).scatter_(
            dim=0, index=torch.tensor(label), value=1
        )
    )

    data_train = datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=one_hot_encoding,
    )
    data_test = datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=one_hot_encoding,
    )
    loader_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)
    return loader_train, loader_test
