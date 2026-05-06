import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset


def load_fashion_mnist_subset(train_size=10000, test_size=2000, random_seed=42):
    """
    Loads Fashion-MNIST, normalizes images, flattens them into vectors,
    and returns smaller train/test subsets for faster experiments.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    rng = np.random.default_rng(random_seed)

    train_indices = rng.choice(len(train_dataset), size=train_size, replace=False)
    test_indices = rng.choice(len(test_dataset), size=test_size, replace=False)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, test_subset


def dataset_to_numpy(dataset):
    """
    Converts a PyTorch dataset into NumPy arrays.
    Images are flattened from 28x28 to 784-dimensional vectors.
    """

    X = []
    y = []

    for image, label in dataset:
        image = image.view(-1).numpy()
        X.append(image)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == "__main__":
    train_subset, test_subset = load_fashion_mnist_subset()

    X_train, y_train = dataset_to_numpy(train_subset)
    X_test, y_test = dataset_to_numpy(test_subset)

    print("Fashion-MNIST subset loaded successfully.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)