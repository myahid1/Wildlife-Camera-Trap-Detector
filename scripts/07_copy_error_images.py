import shutil
from pathlib import Path

import pandas as pd


def main():
    input_path = Path("outputs/metrics/error_analysis.csv")
    output_dir = Path("outputs/error_examples")

    df = pd.read_csv(input_path)

    for _, row in df.iterrows():
        image_path = Path(row["image_path"])
        error_type = row["error_type"]

        destination_folder = output_dir / error_type
        destination_folder.mkdir(parents=True, exist_ok=True)

        if image_path.exists():
            destination_path = destination_folder / image_path.name
            shutil.copy(image_path, destination_path)
        else:
            print(f"Image not found: {image_path}")

    print(f"Copied error images to: {output_dir}")


if __name__ == "__main__":
    main()