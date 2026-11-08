from typing import Optional
 
import cv2
import mediapipe as mp
import numpy as np
 
# Constants

NUM_HAND_LANDMARKS = 21
HAND_WRIST_INDEX = 0
 
FACE_NOSE_TIP_INDEX = 1
EYEBROW_INDICES = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)
MOUTH_INDICES = (0, 17, 61, 291)
FACE_INDICES = EYEBROW_INDICES + MOUTH_INDICES  # 14 landmarks
 
NUM_FACE_LANDMARKS_USED = len(FACE_INDICES)  # 14
HAND_FEATURE_DIM = NUM_HAND_LANDMARKS * 3  # 63
FACE_FEATURE_DIM = NUM_FACE_LANDMARKS_USED * 3  # 42
FEATURE_DIM = HAND_FEATURE_DIM + FACE_FEATURE_DIM  # 105
 
mp_holistic = mp.solutions.holistic
 
 
# Holistic helpers
 
def create_holistic(
    static_image_mode: bool = False,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create a configured MediaPipe Holistic instance (usable as a context manager)."""
    return mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=not static_image_mode,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
 
 
# Normalization
 
def normalize_relative(points: np.ndarray, anchor_index: int) -> np.ndarray:
    """
    Make a landmark set position-independent by subtracting an anchor landmark.
 
    Args:
        points: array of shape (N, 3) holding x, y, z per landmark.
        anchor_index: index of the landmark that becomes the origin.
 
    Returns:
        Array of shape (N, 3) where the anchor landmark sits at (0, 0, 0).
    """
    points = np.asarray(points, dtype=np.float32)
    return points - points[anchor_index]
  
# Extraction
 
def _landmarks_to_array(landmark_list) -> np.ndarray:
    """Convert a MediaPipe landmark list to an (N, 3) float32 array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark],
        dtype=np.float32,
    )
 
 
def extract_hand_vector(results) -> Optional[np.ndarray]:
    """Return the 63-value wrist-relative right-hand vector, or None if no right hand."""
    if results.right_hand_landmarks is None:
        return None
 
    hand = _landmarks_to_array(results.right_hand_landmarks)
    if hand.shape[0] != NUM_HAND_LANDMARKS:
        return None
 
    return normalize_relative(hand, HAND_WRIST_INDEX).flatten()
 
 
def extract_face_vector(results) -> Optional[np.ndarray]:
    """Return the 42-value nose-relative face vector, or None if no face detected."""
    if results.face_landmarks is None:
        return None
 
    face = _landmarks_to_array(results.face_landmarks)
    if face.shape[0] <= max(FACE_INDICES):
        return None
 
    # Normalize against the nose tip using the full mesh, then slice the 14 points.
    face_relative = normalize_relative(face, FACE_NOSE_TIP_INDEX)
    return face_relative[list(FACE_INDICES)].flatten()
 
 
def extract_features(results, require_face: bool = False) -> Optional[np.ndarray]:
    """
    Build the 105-float feature vector from a MediaPipe Holistic result.
 
    Args:
        results: the object returned by ``holistic.process(rgb_image)``.
        require_face: if True, return None when no face is detected. If False
            (default), a missing face is filled with 42 zeros so hand-only
            images (e.g. cropped ASL dataset photos) remain usable.
 
    Returns:
        np.ndarray of shape (105,) and dtype float32, or None if the right
        hand is missing (or the face is missing while ``require_face`` is True).
    """
    hand_vec = extract_hand_vector(results)
    if hand_vec is None:
        return None
 
    face_vec = extract_face_vector(results)
    if face_vec is None:
        if require_face:
            return None
        face_vec = np.zeros(FACE_FEATURE_DIM, dtype=np.float32)
 
    features = np.concatenate([hand_vec, face_vec]).astype(np.float32)
    assert features.shape[0] == FEATURE_DIM
    return features
 
 
def create_hands(static_image_mode: bool = True, min_detection_confidence: float = 0.5):
    """
    Create a MediaPipe Hands instance (usable as a context manager).
 
    Holistic is pose-driven and finds nothing in close-up hand photos with no
    body in frame. Hands detects the hand directly and uses the same 21-point
    layout, so its output is compatible with the Holistic hand vector.
    """
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    )
 

HAND_DETECTION_ATTEMPTS = (
    (0.0, 0),
    (0.0, 256),
    (0.3, 0),
    (0.3, 256),
    (0.6, 320),
)

def _detect_hand_relative(hands, rgb_image: np.ndarray) -> Optional[np.ndarray]:
    height, width = rgb_image.shape[:2]

    for pad_ratio, target_side in HAND_DETECTION_ATTEMPTS:
        image = rgb_image
        pad = int(round(pad_ratio * max(height, width)))
        if pad > 0:
            image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        if target_side and max(image.shape[:2]) < target_side:
            scale = target_side / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        results = hands.process(image)
        if not results.multi_hand_landmarks:
            continue

        hand = _landmarks_to_array(results.multi_hand_landmarks[0])
        if hand.shape[0] != NUM_HAND_LANDMARKS:
            continue

        relative = normalize_relative(hand, HAND_WRIST_INDEX)
        padded_w = width + 2 * pad
        padded_h = height + 2 * pad
        relative[:, 0] *= padded_w / width
        relative[:, 1] *= padded_h / height
        relative[:,2] *= padded_w / width
        return relative 
 
def extract_hand_only_features(hands, rgb_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Fallback extraction using the Hands model. Returns a 105-float vector with the
    wrist-relative hand values and 42 zeros for the face block, or None if no hand.
    """
    relative = _detect_hand_relative(hands, rgb_image)
    if relative is None:
        return None
 
    hand_vec = relative.astype(np.float32).flatten()
    face_vec = np.zeros(FACE_FEATURE_DIM, dtype=np.float32)
    return np.concatenate([hand_vec, face_vec]).astype(np.float32)
 
 
def process_frame(holistic, bgr_frame: np.ndarray, require_face: bool = False) -> Optional[np.ndarray]:
    """
    Convenience wrapper for live webcam frames: BGR -> RGB -> Holistic -> features.
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    return extract_features(results, require_face=require_face)