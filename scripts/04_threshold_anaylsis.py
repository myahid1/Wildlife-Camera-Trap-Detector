from pathlib import Path

import pandas as pd


def map_true_label(label: str) -> str:
    if label == "car":
        return "vehicle"

    if label == "empty":
        return "empty"

    return "animal"


def calculate_binary_metrics(df: pd.DataFrame) -> dict:
    """
    Binary task:
    animal present vs empty

    For this task:
    - animal and vehicle count as detected objects
    - empty means no object detected
    """
    df["true_object_present"] = df["true_class"].isin(["animal", "vehicle"])
    df["pred_object_present"] = df["best_category"].isin(["animal", "vehicle"])

    true_positive = (
        (df["true_object_present"] == True)
        & (df["pred_object_present"] == True)
    ).sum()

    false_positive = (
        (df["true_object_present"] == False)
        & (df["pred_object_present"] == True)
    ).sum()

    false_negative = (
        (df["true_object_present"] == True)
        & (df["pred_object_present"] == False)
    ).sum()

    true_negative = (
        (df["true_object_present"] == False)
        & (df["pred_object_present"] == False)
    ).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / len(df)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def main():
    input_path = Path("outputs/predictions/megadetector_results.json")
    output_path = Path("outputs/metrics/threshold_analysis.csv")

    import json

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    categories = data["categories"]

    rows = []

    for threshold in [0.3, 0.5, 0.7, 0.9]:
        image_rows = []

        for item in data["annotations"]:
            image_path = item["img_id"]
            filename = image_path.replace("\\", "/").split("/")[-1]
            true_label = filename.split("__")[0]
            true_class = map_true_label(true_label)

            best_category = "empty"
            best_confidence = 0

            for category_id, confidence in zip(item["category"], item["confidence"]):
                if confidence >= threshold and confidence > best_confidence:
                    best_category = categories[str(category_id)]
                    best_confidence = confidence

            image_rows.append(
                {
                    "image_path": image_path,
                    "true_class": true_class,
                    "best_category": best_category,
                    "best_confidence": best_confidence,
                }
            )

        threshold_df = pd.DataFrame(image_rows)
        metrics = calculate_binary_metrics(threshold_df)

        metrics["threshold"] = threshold
        rows.append(metrics)

    results_df = pd.DataFrame(rows)
    results_df = results_df[
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nThreshold analysis:")
    print(results_df)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()