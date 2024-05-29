from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from cv2.typing import MatLike

# Initialize mediapipe hands module
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image: MatLike, model) -> tuple[MatLike, NamedTuple]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image: MatLike, results: NamedTuple) -> None:
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(image: MatLike) -> np.ndarray:
    # Detect landmarks in image
    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
        _, results = mediapipe_detection(image, holistic)

    # Extract landmarks from results
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([left_hand, right_hand])


def track_hands(winwidth: int, winheight: int) -> None:
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)

            draw_landmarks(image, results)

            image = cv2.resize(image, (winwidth, winheight))

            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vidcap.release()
    cv2.destroyAllWindows()
