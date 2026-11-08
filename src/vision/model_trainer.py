import logging
import math
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CSV = PROJECT_ROOT / "data" / "asl_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "asl_classifier.plk"

TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_trainer")

def load_dataset(csv_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run `python -m src.vision.dataset_generator` first."
        )
 
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV is missing the required 'label' column.")
 
    df = df.dropna()
    if df.empty:
        raise ValueError("Dataset is empty after removing incomplete rows.")
 
    labels = df["label"].astype(str)
    features = df.drop(columns=["label"])
    return features, labels

def train_model() -> bool:
    """Train and save the classifier. Returns True on success."""
    try:
        features, labels = load_dataset(DATASET_CSV)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return False
 
    logger.info("Loaded %d samples, %d features, %d classes.", len(features), features.shape[1], labels.nunique())
 
    class_count = labels.nunique()
    test_count = math.ceil(len(labels) * TEST_SIZE)
    if labels.value_counts().min() < 2 or test_count < class_count:
        logger.error(
            "Not enough samples for a stratified train/test split. "
            "Add more images so every class has at least 2 samples and the test split "
            "contains at least one sample per class."
        )
        return False
 
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    logger.info("Train samples: %d | Test samples: %d", len(x_train), len(x_test))
 
    logger.info("Training linear SVC...")
    model = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    model.fit(x_train.values, y_train)
 
    predictions = model.predict(x_test.values)
    accuracy = accuracy_score(y_test, predictions) * 100.0
 
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Test accuracy: {accuracy:.2f}%")
    print("\nPer-letter report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("=" * 50)
 
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    return True

if __name__ == "__main__":
    sys.exit(0 if train_model() else 1)