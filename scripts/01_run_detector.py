from pathlib import Path

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils


def main():
    image_dir = Path("data/raw/sample_images")
    output_image_dir = Path("outputs/figures")
    output_json_path = Path("outputs/predictions/megadetector_results.json")

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    image_files = list(image_dir.glob("*.*"))

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No images found in {image_dir}. Put some .jpg images there first."
        )

    print(f"Found {len(image_files)} images.")
    print("Loading MegaDetectorV6 on CPU...")

    model = pw_detection.MegaDetectorV6(
        device="cpu",
        pretrained=True,
        version="MDV6-yolov10-c"
    )

    print("Running detection...")

    results = model.batch_image_detection(
        str(image_dir),
        batch_size=1
    )

    print("Saving boxed images...")

    pw_utils.save_detection_images(
        results,
        str(output_image_dir),
        str(image_dir),
        overwrite=True
    )

    print("Saving JSON predictions...")

    pw_utils.save_detection_json(
        results,
        str(output_json_path),
        categories=model.CLASS_NAMES,
        exclude_category_ids=[],
        exclude_file_path=None
    )

    print("\nDone.")
    print(f"Boxed images saved to: {output_image_dir}")
    print(f"Prediction JSON saved to: {output_json_path}")


if __name__ == "__main__":
    main()