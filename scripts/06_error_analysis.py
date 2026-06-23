from pathlib import Path

import pandas as pd


def map_true_label(label: str) -> str:
    if label == "car":
        return "vehicle"

    if label == "empty":
        return "empty"

    return "animal"


def main():
    input_path = Path("outputs/metrics/prediction_summary.csv")
    output_path = Path("outputs/metrics/error_analysis.csv")

    df = pd.read_csv(input_path)

    df["true_class"] = df["true_label_from_filename"].apply(map_true_label)
    df["correct"] = df["true_class"] == df["best_category"]

    errors = df[df["correct"] == False].copy()

    def get_error_type(row):
        if row["true_class"] == "empty" and row["best_category"] != "empty":
            return "false_positive"

        if row["true_class"] != "empty" and row["best_category"] == "empty":
            return "false_negative"

        return "wrong_class"

    errors["error_type"] = errors.apply(get_error_type, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_path, index=False)

    print("\nError Analysis")
    print("--------------")
    print(f"Total images: {len(df)}")
    print(f"Correct images: {df['correct'].sum()}")
    print(f"Incorrect images: {len(errors)}")

    print("\nError types:")
    print(errors["error_type"].value_counts())

    print("\nErrors:")
    print(
        errors[
            [
                "image_path",
                "true_class",
                "best_category",
                "best_confidence",
                "error_type",
            ]
        ]
    )

    print(f"\nSaved error analysis to: {output_path}")


if __name__ == "__main__":
    main()