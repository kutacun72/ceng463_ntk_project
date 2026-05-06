import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_fashion_mnist_subset


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


def train_one_epoch(model, train_loader, criterion, optimizer, device):
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

    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
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

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    return accuracy, macro_f1


def run_experiment(hidden_size, train_loader, test_loader, device, num_epochs=10):
    model = MLP(
        input_size=784,
        hidden_size=hidden_size,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    print("\n" + "=" * 60)
    print(f"Training MLP with hidden size = {hidden_size}")
    print("=" * 60)

    final_accuracy = 0.0
    final_macro_f1 = 0.0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        accuracy, macro_f1 = evaluate(
            model=model,
            test_loader=test_loader,
            device=device
        )

        final_accuracy = accuracy
        final_macro_f1 = macro_f1

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.4f} "
            f"Macro-F1: {macro_f1:.4f}"
        )

    runtime = time.time() - start_time

    return {
        "Model": f"MLP hidden={hidden_size}",
        "Hidden Size": hidden_size,
        "Accuracy": round(final_accuracy, 4),
        "Macro-F1": round(final_macro_f1, 4),
        "Runtime (s)": round(runtime, 2),
        "Epochs": num_epochs,
        "Learning Rate": 0.001
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_subset, test_subset = load_fashion_mnist_subset(
        train_size=10000,
        test_size=2000,
        random_seed=42
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=64,
        shuffle=False
    )

    hidden_sizes = [64, 128, 256, 512]
    results = []

    for hidden_size in hidden_sizes:
        result = run_experiment(
            hidden_size=hidden_size,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=10
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("MLP Width Ablation Results")
    print("=" * 60)
    print(results_df)

    results_df.to_csv("results/mlp_width_ablation_results.csv", index=False)
    print("\nResults saved to results/mlp_width_ablation_results.csv")


if __name__ == "__main__":
    main()