import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader

from data_loader import load_fashion_mnist_subset, dataset_to_numpy


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_mlp(train_subset, test_subset, device, num_epochs=10):
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    model = MLP(hidden_size=256).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"MLP Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss / len(train_loader):.4f}")

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_predictions)


def main():
    start_time = time.time()

    train_subset, test_subset = load_fashion_mnist_subset(
        train_size=10000,
        test_size=2000,
        random_seed=42
    )

    X_train, y_train = dataset_to_numpy(train_subset)
    X_test, y_test = dataset_to_numpy(test_subset)

    print("Training RBF SVM...")
    svm_model = SVC(kernel="rbf", C=10, gamma="scale")
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_macro_f1 = f1_score(y_test, svm_predictions, average="macro")

    print("RBF SVM Accuracy:", round(svm_accuracy, 4))
    print("RBF SVM Macro-F1:", round(svm_macro_f1, 4))

    svm_cm = confusion_matrix(y_test, svm_predictions)

    plot_confusion_matrix(
        cm=svm_cm,
        title="Confusion Matrix - RBF SVM",
        save_path="figures/confusion_matrix_rbf_svm.png"
    )

    print("Training MLP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_labels, mlp_predictions = train_mlp(
        train_subset=train_subset,
        test_subset=test_subset,
        device=device,
        num_epochs=10
    )

    mlp_accuracy = accuracy_score(true_labels, mlp_predictions)
    mlp_macro_f1 = f1_score(true_labels, mlp_predictions, average="macro")

    print("MLP Accuracy:", round(mlp_accuracy, 4))
    print("MLP Macro-F1:", round(mlp_macro_f1, 4))

    mlp_cm = confusion_matrix(true_labels, mlp_predictions)

    plot_confusion_matrix(
        cm=mlp_cm,
        title="Confusion Matrix - MLP hidden=256",
        save_path="figures/confusion_matrix_mlp.png"
    )

    runtime = time.time() - start_time
    print("\nConfusion matrices saved:")
    print("figures/confusion_matrix_rbf_svm.png")
    print("figures/confusion_matrix_mlp.png")
    print("Total runtime:", round(runtime, 2), "seconds")


if __name__ == "__main__":
    main()