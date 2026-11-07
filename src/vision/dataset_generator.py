import csv
import logging
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.pipeline import FEATURE_DIM, create_holistic, extract_features

DATASET_DIR = PROJECT_ROOT / "data" / "asl_dataset"
OUTPUT_CSV = PROJECT_ROOT / "data" / "asl_dataset.csv"
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", ".webp"}
LOG_EVERY = 100

# we're gonna have letters, and ASL digits that share a handshape with a letter, so we need to make sure we don't have duplicates
INCLUDE_DIGITS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_generator")

def find_letter(dataset_dir: Path):
    folders = []
    for child in dataset_dir.iterdir():
        if not child.is_dir() or len(child.name) != 1 or not child.name.isalpha():
            continue
        if child.name.isalpha():
            folders.append((child.name.upper(), child))
        elif child.name.isalpha():
            if INCLUDE_DIGITS:
                folders.append((child.name, child))
            else:
                logger.info("Skipping digit folder: '%s' (INCLUDE_DIGITS=False)", child.name)
    return sorted(folders, key=lambda item: item[0])

def list_images(folder: Path):
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

def generate_dataset() -> bool:
    if not DATASET_DIR.is_dir():
        logger.error("Dataset directory does not exist: %s", DATASET_DIR)
        return False

    letter_folders = find_letter(DATASET_DIR)
    if not letter_folders:
        logger.error("No letter folders found in dataset directory: %s", ", 0-9" if INCLUDE_DIGITS else "", DATASET_DIR)
        return False

    logger.info("Found %d letter folders in dataset directory: %s", len(letter_folders), DATASET_DIR)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_failure = 0
    per_letter = {}
    header = ["label"] + [f"f{i}" for i in range(FEATURE_DIM)]

    with open(OUTPUT_CSV, "w", newline="") as csv_file, \
            create_holistic(static_image_mode=True) as holistic:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for letter, folder in letter_folders:
            images = list_images(folder)
            logger.info("Processing letter '%s' with %d images", letter, len(images))
            success = 0
            failure = 0

            for index, image_path in enumerate(images, start=1):
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    logger.warning("Could not read image: %s", image_path)
                    failure += 1
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                try:
                    results = holistic.process(image_rgb)
                    features = extract_features(results)
                except Exception as exc:
                    logger.warning("Error processing image '%s': %s", image_path, exc)
                    failure += 1
                    continue

                if features is None:
                    failure += 1
                else:
                    writer.writerow([letter] + [f"{value:.8f}" for value in features])
                    success += 1

                if index % LOG_EVERY == 0:
                    logger.info("'%s': %d/%d images processed", letter, index, len(images))

            per_letter[letter] = (success, failure)
            total_success += success
            total_failure += failure
            logger.info("Finished '%s': %d successful, %d failed", letter, success, failure)

        total = total_success + total_failure
        rate = (total_success / total) if total else 0.0

        print("\n " + "=" * 40)
        print("Dataset Generation Summary")
        print("=" * 40)
        for letter , (success, failure) in per_letter.items():
            print(f"Letter '{letter}': {success} successful, {failure} failed")
        print("-" * 40)
        print(f"  Total images: {total}")
        print(f"  Total successful: {total_success}")
        print(f"  Total failed: {total_failure}")
        print(f"  Success rate: {rate:.2%}")
        print(f"  Output CSV: {OUTPUT_CSV}")
        print("=" * 40 + "\n")

        if total_success == 0:
            logger.error("No successful feature extractions. Dataset generation failed.")
            return False
    print("Dataset generation completed successfully.")
    return True

if __name__ == "__main__":
    sys.exit(0 if generate_dataset() else 1)
