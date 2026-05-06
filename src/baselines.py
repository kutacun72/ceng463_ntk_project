import time
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_loader import load_fashion_mnist_subset, dataset_to_numpy


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    runtime = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Runtime:", round(runtime, 2), "seconds")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Macro-F1": round(macro_f1, 4),
        "Runtime (s)": round(runtime, 2)
    }


def main():
    train_subset, test_subset = load_fashion_mnist_subset(
        train_size=10000,
        test_size=2000,
        random_seed=42
    )

    X_train, y_train = dataset_to_numpy(train_subset)
    X_test, y_test = dataset_to_numpy(test_subset)

    print("Dataset loaded.")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    models = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                n_jobs=-1
            )
        ),
        (
            "Linear SVM",
            LinearSVC(
                max_iter=5000
            )
        ),
        (
            "RBF SVM",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale"
            )
        )
    ]

    results = []

    for model_name, model in models:
        result = evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("Final Baseline Results")
    print("=" * 60)
    print(results_df)

    results_df.to_csv("results/baseline_results.csv", index=False)
    print("\nResults saved to results/baseline_results.csv")


if __name__ == "__main__":
    main()