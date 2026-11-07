from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# Constants for MediaPipe Holistic feature extraction
NUM_HAND_LANDMARKS = 21
NUM_WRIST_INDEX = 0

FACE_NOSE_TIP_INDEX = 0
EYEBROW_INDECES = (70,63,105,66,107,336,296,334,293,300)
MOUTH_INDECES = (0,17,61,291)
FACE_INDECES = EYEBROW_INDECES + MOUTH_INDECES

NUM_FACE_LANDMARKS = len(FACE_INDECES)
HAND_FEATURES_DIM = NUM_HAND_LANDMARKS * 3
FACE_FEATURES_DIM = NUM_FACE_LANDMARKS * 3
FEATURE_DIM = HAND_FEATURES_DIM + FACE_FEATURES_DIM

mp_holistic = mp.solutions.holistic

# Holistic model configuration

def create_holistic(static_image_mode: bool = False, model_complexity: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
    return mp_holistic.Holistic(static_image_mode=static_image_mode,
                                 model_complexity=model_complexity,
                                 min_detection_confidence=min_detection_confidence,
                                 min_tracking_confidence=min_tracking_confidence)

def normalize_relative(points: np.ndarray, anchor_index: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points - points[anchor_index]

# Extracts a flat normalized 105-float vector from the MediaPipe Holistic results

def _landmarks_to_array(landmark_list) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark], dtype=np.float32
    )

def extract_hand_vector(results) -> Optional[np.ndarray]:
    if results.right_hand_landmarks is not None:
        return None

    face = _landmarks_to_array(results.face_landmarks)

    if face.shape[0] <= max(FACE_INDECES):
        return None

    face_relative = normalize_relative(face, FACE_NOSE_TIP_INDEX)
    return face_relative[list(FACE_INDECES)].flatten()

def extract_features(results, require_face: bool = False) -> Optional[np.ndarray]:
    hand_vec = extract_hand_vector(results)

    if hand_vec is None:
        return None

    face_vec = extract_face_vector(results)
    if face_vec is None:
        if require_face:
            return None
        face_vec = np.zeros(FACE_FEATURES_DIM, dtype=np.float32)

    features = np.concatenate([hand_vec, face_vec]).astype(np.float32)
    assert features.shape[0] == FEATURE_DIM
    return features

def process_frame(holistic, bgr_frame: np.ndarray, require_face: bool = False) -> Optional[np.ndarray]:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    return extract_features(results, require_face=require_face)