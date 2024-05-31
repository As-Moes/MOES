
import cv2
import numpy as np

from ..utils import utils_cv
from ..utils import utils_math

from .mpLoader import MediaPipeLoader

from cv2.typing import MatLike
from typing import NamedTuple

#--------------------------------------------------------------------------------------------------------

# Run MediaPipe model inference
def mediapipe_detection(image: MatLike, model) -> tuple[MatLike, NamedTuple]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Draw landmarks in some image
def draw_landmarks(image: MatLike, results: NamedTuple, mp: MediaPipeLoader) -> MatLike:
    mp.mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.mp_holistic.POSE_CONNECTIONS)
    mp.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.mp_holistic.HAND_CONNECTIONS)
    mp.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.mp_holistic.HAND_CONNECTIONS)
    return image

#--------------------------------------------------------------------------------------------------------

# Extract keypoints from both hands using MediaPipe
def extract_keypoints_raw(image: MatLike, mp: MediaPipeLoader, show: bool = False) -> np.ndarray:
    _, results = mediapipe_detection(image, mp.model)
    
    if show:
        image = draw_landmarks(image, results, mp)
        utils_cv.show(image)
    
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([left_hand, right_hand])

# Extract keypoints from both hands using MediaPipe and compute hand angles for each hand
# Extract body/face keypoints and compute distances between desired targets
def extract_angles_distances(image: MatLike, mp: MediaPipeLoader, show: bool = False) -> np.ndarray:
    _, results = mediapipe_detection(image, mp.model)

    if show:
        image = draw_landmarks(image, results, mp)
        utils_cv.show(image)
    
    # CUSTOM FEATURE 1: HANDs ANGLES (26 angles for each hand => 52 angles) 
    angles = []
    for landmarks_list in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if not landmarks_list:  # hand not visible in frame
            angles.extend(list(np.zeros(26)))
        else:
            # Get all hand landmarks points
            # Find the angle between all adjacent vectors 
            hand = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in landmarks_list.landmark]
            hand_vectors = {conn: hand[conn[0]] - hand[conn[1]] for conn in mp.mp_holistic.HAND_CONNECTIONS}
            hand_angles  = [utils_math.angle_between(hand_vectors[vec1_idx], hand_vectors[vec2_idx])
                           for (vec1_idx, vec2_idx) in mp.HAND_ADJACENT_CONNECTIONS]
            angles.extend(hand_angles)
    
    # CUSTOM FEATURE 2: POSE DISTANCES (37 distances)
    distances = np.zeros(37)
    if results.pose_landmarks:
        pose = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in results.pose_landmarks.landmark]

        # Extract max pose distance
        # Normalize all pose landmarks 
        central_pose = (pose[12] + pose[11]) / 2
        dist_max_pose = max([np.linalg.norm(lm - central_pose) for lm in pose])
        pose = np.array([(lm - central_pose) / dist_max_pose for lm in pose])
 
        # Compute target distances
        distances = [np.linalg.norm(pose[idx1] - pose[idx2]) for (idx1, idx2) in mp.TARGET_DISTANCES]
 
    return np.concatenate([angles, distances])

#--------------------------------------------------------------------------------------------------------

# Turn on the webcam and run MediaPipe model inference
# on each frame and draw keypoints
def live_hands_tracking(window_size: tuple, mp: MediaPipeLoader) -> None:
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret: break
    
        image, results = mediapipe_detection(frame, mp.model)
        image          = draw_landmarks(image, results, mp)
        image          = utils_cv.resize_to(image, window_size)
        
        cv2.imshow('Live', image)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
    vidcap.release()
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------------
