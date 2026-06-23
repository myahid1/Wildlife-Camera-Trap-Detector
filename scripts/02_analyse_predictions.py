import argparse
import json
from pathlib import Path

import pandas as pd


def get_label_from_filename(image_path: str) -> str:
    """
    Example:
    data/raw/sample_images/cat__abc123.jpg -> cat
    """
    filename = image_path.replace("\\", "/").split("/")[-1]
    return filename.split("__")[0]


def main(predictions_path: Path, output_path: Path, threshold: float) -> None:
    with open(predictions_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    categories = data["categories"]
    rows = []

    for item in data["annotations"]:
        image_path = item["img_id"]
        true_label = get_label_from_filename(image_path)

        boxes = item["bbox"]
        category_ids = item["category"]
        confidences = item["confidence"]

        kept_detections = []

        for box, category_id, confidence in zip(boxes, category_ids, confidences):
            if confidence >= threshold:
                category_name = categories[str(category_id)]

                kept_detections.append(
                    {
                        "box": box,
                        "category": category_name,
                        "confidence": confidence,
                    }
                )

        if len(kept_detections) == 0:
            best_category = "empty"
            best_confidence = 0
        else:
            best_detection = max(
                kept_detections,
                key=lambda detection: detection["confidence"]
            )
            best_category = best_detection["category"]
            best_confidence = best_detection["confidence"]

        rows.append(
            {
                "image_path": image_path,
                "true_label_from_filename": true_label,
                "num_detections": len(kept_detections),
                "best_category": best_category,
                "best_confidence": round(best_confidence, 4),
                "animal_detected": best_category == "animal",
                "vehicle_detected": best_category == "vehicle",
                "person_detected": best_category == "person",
            }
        )

    df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nPrediction summary")
    print("------------------")
    print(f"Images analysed: {len(df)}")
    print(f"Threshold used: {threshold}")
    print(f"Images with detections: {(df['num_detections'] > 0).sum()}")
    print(f"Images predicted empty: {(df['num_detections'] == 0).sum()}")

    print("\nBest category counts:")
    print(df["best_category"].value_counts())

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/predictions/megadetector_results.json"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/prediction_summary.csv"),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    main(
        predictions_path=args.predictions,
        output_path=args.output,
        threshold=args.threshold,
    )