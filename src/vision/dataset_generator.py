"""
dataset_generator.py - Build data/asl_dataset.csv from an image dataset.
 
Expects an unzipped dataset at data/asl_dataset/ with one subfolder per letter:
 
    data/asl_dataset/A/*.jpg
    data/asl_dataset/B/*.jpg
    ...
 
Each image is read with OpenCV, converted to RGB, run through MediaPipe
Holistic (static_image_mode=True) and converted to a 105-float feature vector
via pipeline.extract_features. Successful vectors are written to the CSV with
their letter label.
 
Run from the project root:
    python -m src.vision.dataset_generator
"""
 
import csv
import logging
import sys
from pathlib import Path
 
import cv2
 
# Allow running both as a module and as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
 
from src.vision.pipeline import (  # noqa: E402
    FEATURE_DIM,
    create_hands,
    create_holistic,
    extract_features,
    extract_hand_only_features,
)
 
DATASET_DIR = PROJECT_ROOT / "data" / "asl_dataset"
OUTPUT_CSV = PROJECT_ROOT / "data" / "asl_dataset.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOG_EVERY = 100  # progress message every N images
 
# SignRacer types letters, and several ASL digits share a handshape with a letter
# (0/O, 2/V, 6/W, ...), which hurts letter accuracy. Set to True to also train on 0-9.
INCLUDE_DIGITS = False
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_generator")
 
 
def find_letter_folders(dataset_dir: Path):
    """
    Return a sorted list of (LABEL, folder_path) for single-character subfolders.
 
    Letter folders (a-z or A-Z) are always included and labelled in uppercase.
    Digit folders (0-9) are included only when INCLUDE_DIGITS is True.
    """
    folders = []
    for child in dataset_dir.iterdir():
        if not child.is_dir() or len(child.name) != 1 or not child.name.isascii():
            continue
        if child.name.isalpha():
            folders.append((child.name.upper(), child))
        elif child.name.isdigit():
            if INCLUDE_DIGITS:
                folders.append((child.name, child))
            else:
                logger.info("Skipping digit folder '%s' (INCLUDE_DIGITS is False).", child.name)
    return sorted(folders, key=lambda item: item[0])
 
 
def list_images(folder: Path):
    """Return a sorted list of image files in a folder."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
 
 
def generate_dataset() -> bool:
    """Process the whole dataset and write the CSV. Returns True on success."""
    if not DATASET_DIR.is_dir():
        logger.error("Dataset directory not found: %s", DATASET_DIR)
        return False
 
    letter_folders = find_letter_folders(DATASET_DIR)
    if not letter_folders:
        logger.error("No class subfolders (a-z%s) found in %s", ", 0-9" if INCLUDE_DIGITS else "", DATASET_DIR)
        return False
 
    logger.info("Found %d class folders: %s", len(letter_folders), " ".join(l for l, _ in letter_folders))
 
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
 
    total_success = 0
    total_failure = 0
    per_letter = {}
    header = ["label"] + [f"f{i}" for i in range(FEATURE_DIM)]
 
    fallback_used = 0
 
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file, \
            create_holistic(static_image_mode=True) as holistic, \
            create_hands(static_image_mode=True) as hands:
        writer = csv.writer(csv_file)
        writer.writerow(header)
 
        for letter, folder in letter_folders:
            images = list_images(folder)
            logger.info("Processing letter '%s' (%d images)...", letter, len(images))
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
                    if features is None:
                        features = extract_hand_only_features(hands, image_rgb)
                        if features is not None:
                            fallback_used += 1
                except Exception as exc:  # keep going if a single image misbehaves
                    logger.warning("Error processing %s: %s", image_path, exc)
                    failure += 1
                    continue
 
                if features is None:
                    failure += 1
                else:
                    writer.writerow([letter] + [f"{value:.8f}" for value in features])
                    success += 1
 
                if index % LOG_EVERY == 0:
                    logger.info("  '%s': %d/%d images processed", letter, index, len(images))
 
            per_letter[letter] = (success, failure)
            total_success += success
            total_failure += failure
            logger.info("Finished '%s': %d succeeded, %d failed", letter, success, failure)
 
    total = total_success + total_failure
    rate = (total_success / total * 100.0) if total else 0.0
 
    print("\n" + "=" * 50)
    print("DATASET GENERATION SUMMARY")
    print("=" * 50)
    for letter, (success, failure) in per_letter.items():
        print(f"  {letter}: {success:5d} ok | {failure:5d} failed")
    print("-" * 50)
    print(f"  Total images : {total}")
    print(f"  Successful   : {total_success}")
    print(f"  Failed       : {total_failure}")
    print(f"  Via Hands fallback: {fallback_used}")
    print(f"  Success rate : {rate:.2f}%")
    print(f"  Output file  : {OUTPUT_CSV}")
    print("=" * 50)
 
    if total_success == 0:
        print("FAILURE: no feature vectors were extracted.")
        return False
 
    print("SUCCESS: dataset written.")
    return True
 
 
if __name__ == "__main__":
    sys.exit(0 if generate_dataset() else 1)
 