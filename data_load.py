import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets

def load_cifar10(data_dir='./data_cifar10', train_split=0.8):
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

    X_train = np.array([np.transpose(img.numpy(), (1, 2, 0)).flatten() for img, _ in train_set])
    y_train = np.array([label for _, label in train_set])
    X_test = np.array([np.transpose(img.numpy(), (1, 2, 0)).flatten() for img, _ in test_set])
    y_test = np.array([label for _, label in test_set])

    # Normalize to [-1, 1]
    X_train = (X_train - 0.5) / 0.5
    X_test = (X_test - 0.5) / 0.5

    split_index = int(train_split * len(X_train))
    return X_train[:split_index], y_train[:split_index], X_train[split_index:], y_train[split_index:], X_test, y_test