import pandas as pd


def main():
    baseline_path = "results/baseline_results.csv"
    mlp_path = "results/mlp_results.csv"
    ablation_path = "results/mlp_width_ablation_results.csv"

    baseline_df = pd.read_csv(baseline_path)
    mlp_df = pd.read_csv(mlp_path)
    ablation_df = pd.read_csv(ablation_path)

    # For the main comparison table, use baseline models + single MLP result
    main_results = pd.concat(
        [baseline_df, mlp_df[["Model", "Accuracy", "Macro-F1", "Runtime (s)"]]],
        ignore_index=True
    )

    print("\n" + "=" * 60)
    print("Main Model Comparison Results")
    print("=" * 60)
    print(main_results)

    main_results.to_csv("results/main_model_comparison.csv", index=False)

    print("\nMain comparison saved to:")
    print("results/main_model_comparison.csv")

    print("\n" + "=" * 60)
    print("MLP Width Ablation Results")
    print("=" * 60)
    print(ablation_df)

    print("\nAblation results already saved to:")
    print("results/mlp_width_ablation_results.csv")


if __name__ == "__main__":
    main()