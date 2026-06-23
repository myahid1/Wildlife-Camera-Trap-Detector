from pathlib import Path

import pandas as pd


def map_true_label(label: str) -> str:
    """
    Convert filename label into MegaDetector-style class.

    Examples:
    cat, deer, coyote -> animal
    car -> vehicle
    empty -> empty
    """
    if label == "car":
        return "vehicle"

    if label == "empty":
        return "empty"

    return "animal"


def calculate_metrics(df: pd.DataFrame, target_class: str) -> dict:
    true_positive = (
        (df["true_class"] == target_class)
        & (df["best_category"] == target_class)
    ).sum()

    false_positive = (
        (df["true_class"] != target_class)
        & (df["best_category"] == target_class)
    ).sum()

    false_negative = (
        (df["true_class"] == target_class)
        & (df["best_category"] != target_class)
    ).sum()

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "class": target_class,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def main():
    input_path = Path("outputs/metrics/prediction_summary.csv")
    output_path = Path("outputs/metrics/evaluation_metrics.csv")

    df = pd.read_csv(input_path)

    df["true_class"] = df["true_label_from_filename"].apply(map_true_label)
    df["correct"] = df["true_class"] == df["best_category"]

    accuracy = df["correct"].mean()

    print("\nEvaluation Results")
    print("------------------")
    print(f"Total images: {len(df)}")
    print(f"Correct predictions: {df['correct'].sum()}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion matrix:")
    confusion_matrix = pd.crosstab(
        df["true_class"],
        df["best_category"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )
    print(confusion_matrix)

    metrics = [
        calculate_metrics(df, "animal"),
        calculate_metrics(df, "vehicle"),
        calculate_metrics(df, "empty"),
    ]

    metrics_df = pd.DataFrame(metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)

    print("\nClass metrics:")
    print(metrics_df)

    print(f"\nSaved evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()