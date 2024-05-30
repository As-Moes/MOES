from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from cv2.typing import MatLike

# Initialize mediapipe hands module
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

HAND_ADJACENT_CONNECTIONS = [
    ((0, 1), (1, 2)),
    ((0, 5), (0, 1)),
    ((0, 5), (0, 17)),
    ((0, 5), (5, 6)),
    ((0, 5), (5, 9)),
    ((0, 17), (0, 1)),
    ((0, 17), (13, 17)),
    ((1, 2), (2, 3)),
    ((3, 4), (2, 3)),
    ((5, 6), (5, 9)),
    ((5, 6), (6, 7)),
    ((5, 9), (9, 10)),
    ((5, 9), (9, 13)),
    ((6, 7), (7, 8)),
    ((9, 10), (9, 13)),
    ((9, 10), (10, 11)),
    ((10, 11), (11, 12)),
    ((13, 14), (9, 13)),
    ((13, 14), (13, 17)),
    ((13, 14), (14, 15)),
    ((13, 17), (9, 13)),
    ((14, 15), (15, 16)),
    ((17, 18), (0, 17)),
    ((17, 18), (13, 17)),
    ((17, 18), (18, 19)),
    ((18, 19), (19, 20))
]


# def find_adjacent_connections():
#     adjacent = []
#     for a in mp_holistic.HAND_CONNECTIONS:
#         for b in mp_holistic.HAND_CONNECTIONS:
#             if a != b and (a[0] in b or a[1] in b) and (b, a) not in adjacent:
#                 adjacent.append((a, b))
#     return adjacent
# HAND_ADJACENT_CONNECTIONS = sorted(find_adjacent_connections())


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
