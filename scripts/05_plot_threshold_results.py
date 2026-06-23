from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    input_path = Path("outputs/metrics/threshold_analysis.csv")
    output_path = Path("outputs/figures/threshold_analysis.png")

    df = pd.read_csv(input_path)

    plt.figure(figsize=(8, 5))

    plt.plot(df["threshold"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    plt.plot(df["threshold"], df["recall"], marker="o", label="Recall")
    plt.plot(df["threshold"], df["f1_score"], marker="o", label="F1-score")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title("MegaDetector Performance Across Confidence Thresholds")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved chart to: {output_path}")


if __name__ == "__main__":
    main()