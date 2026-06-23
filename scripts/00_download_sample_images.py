import argparse
import json
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm


METADATA_ZIP_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/labels/caltech_camera_traps.json.zip"
)

AZURE_IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/"
    "lila-wildlife/caltech-unzipped/cct_images"
)


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as file, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {output_path.name}",
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def load_metadata(metadata_zip_path: Path, extracted_dir: Path) -> dict:
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(metadata_zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)

    json_files = list(extracted_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError("No JSON metadata file found after extraction.")

    with open(json_files[0], "r", encoding="utf-8") as file:
        return json.load(file)


def get_group_from_label(label: str) -> str:
    """
    Convert Caltech Camera Traps label into our evaluation group.

    MegaDetector predicts:
    - animal
    - person
    - vehicle

    We also use:
    - empty
    """
    if label == "empty":
        return "empty"

    if label == "car":
        return "vehicle"

    return "animal"


def clear_folder(folder: Path) -> None:
    if not folder.exists():
        return

    for file_path in folder.glob("*"):
        if file_path.is_file():
            file_path.unlink()


def main(
    n_animals: int,
    n_empty: int,
    n_vehicles: int,
    output_dir: Path,
    clear_output: bool,
) -> None:
    metadata_dir = Path("data/annotations")
    metadata_zip_path = metadata_dir / "caltech_camera_traps.json.zip"
    extracted_metadata_dir = metadata_dir / "caltech_metadata"

    print("Loading metadata...")

    download_file(METADATA_ZIP_URL, metadata_zip_path)
    metadata = load_metadata(metadata_zip_path, extracted_metadata_dir)

    images = metadata["images"]
    annotations = metadata["annotations"]
    categories = metadata["categories"]

    category_id_to_name = {
        category["id"]: category["name"]
        for category in categories
    }

    image_id_to_label = {}

    for annotation in annotations:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        label = category_id_to_name[category_id]
        image_id_to_label[image_id] = label

    target_counts = {
        "animal": n_animals,
        "empty": n_empty,
        "vehicle": n_vehicles,
    }

    selected_counts = {
        "animal": 0,
        "empty": 0,
        "vehicle": 0,
    }

    selected_images = []

    for image in images:
        image_id = image["id"]
        label = image_id_to_label.get(image_id)

        if label is None:
            continue

        group = get_group_from_label(label)

        if group not in target_counts:
            continue

        if selected_counts[group] >= target_counts[group]:
            continue

        selected_images.append(
            {
                "image": image,
                "label": label,
                "group": group,
            }
        )

        selected_counts[group] += 1

        if selected_counts == target_counts:
            break

    print("\nSelected image counts:")
    for group, count in selected_counts.items():
        print(f"{group}: {count}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if clear_output:
        print(f"\nClearing old images from: {output_dir}")
        clear_folder(output_dir)

    print("\nDownloading selected images...")

    for item in tqdm(selected_images, desc="Downloading sample images"):
        image = item["image"]
        label = item["label"]
        group = item["group"]

        file_name = image["file_name"]

        encoded_file_name = quote(file_name, safe="/")
        image_url = f"{AZURE_IMAGE_BASE_URL}/{encoded_file_name}"

        safe_file_name = file_name.replace("/", "__")

        output_path = output_dir / f"{label}__{safe_file_name}"

        try:
            download_file(image_url, output_path)
        except Exception as error:
            print(f"Failed to download {file_name}: {error}")

    print(f"\nDone. Images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_animals", type=int, default=30)
    parser.add_argument("--n_empty", type=int, default=20)
    parser.add_argument("--n_vehicles", type=int, default=10)

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/raw/sample_images"),
    )

    parser.add_argument(
        "--clear_output",
        action="store_true",
        help="Delete old images in the output folder before downloading new ones.",
    )

    args = parser.parse_args()

    main(
        n_animals=args.n_animals,
        n_empty=args.n_empty,
        n_vehicles=args.n_vehicles,
        output_dir=args.output_dir,
        clear_output=args.clear_output,
    )