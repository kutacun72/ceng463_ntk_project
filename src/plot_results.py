import pandas as pd
import matplotlib.pyplot as plt


def plot_main_comparison():
    df = pd.read_csv("results/main_model_comparison.csv")

    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df["Accuracy"])
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Main Model Comparison on Fashion-MNIST")
    plt.ylim(0.75, 0.90)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig("figures/main_model_comparison_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df["Macro-F1"])
    plt.xlabel("Model")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 Comparison on Fashion-MNIST")
    plt.ylim(0.75, 0.90)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig("figures/main_model_comparison_macro_f1.png", dpi=300)
    plt.close()


def plot_width_ablation():
    df = pd.read_csv("results/mlp_width_ablation_results.csv")

    plt.figure(figsize=(7, 5))
    plt.plot(df["Hidden Size"], df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(df["Hidden Size"], df["Macro-F1"], marker="o", label="Macro-F1")
    plt.xlabel("Hidden Layer Width")
    plt.ylabel("Score")
    plt.title("MLP Width Ablation on Fashion-MNIST")
    plt.ylim(0.82, 0.86)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig("figures/mlp_width_ablation.png", dpi=300)
    plt.close()


def main():
    plot_main_comparison()
    plot_width_ablation()

    print("Figures saved successfully:")
    print("figures/main_model_comparison_accuracy.png")
    print("figures/main_model_comparison_macro_f1.png")
    print("figures/mlp_width_ablation.png")


if __name__ == "__main__":
    main()