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

DESIRED_DISTANCES = [
    (0, 13),
    (0, 14),
    (0, 15),
    (0, 16),
    (11, 13),
    (11, 14),
    (11, 15),
    (11, 16),
    (12, 13),
    (12, 14),
    (12, 15),
    (12, 16),
    (13, 14),
    (15, 16),
    (15, 17),
    (15, 18),
    (15, 19),
    (15, 20),
    (16, 17),
    (16, 18),
    (16, 19),
    (16, 20),
    (17, 18),
    (17, 19),
    (17, 20),
    (17, 21),
    (17, 22),
    (18, 19),
    (18, 20),
    (18, 21),
    (18, 22),
    (19, 20),
    (19, 21),
    (19, 22),
    (20, 21),
    (20, 22),
    (21, 22)
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
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)


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


def angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    u_vec1 = vec1 / np.linalg.norm(vec1)  # unit vector
    u_vec2 = vec2 / np.linalg.norm(vec2)
    cos = np.dot(u_vec1, u_vec2)
    return np.arccos(cos)


def extract_custom_features(image: MatLike) -> np.ndarray:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        _, results = mediapipe_detection(image, holistic)

    # CUSTOM FEATURE 1: HAND ANGLES
    angles = []
    for landmarks_list in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if not landmarks_list:  # hand not visible in frame
            angles.extend(list(np.zeros(26)))
        else:
            hand = [np.array([landmark.x, landmark.y, landmark.z])
                    for landmark in landmarks_list.landmark]

            # Compute connections (vectors)
            hand_vectors = {conn: hand[conn[0]] - hand[conn[1]]
                            for conn in mp_holistic.HAND_CONNECTIONS}

            # Find the angle between all adjacent vectors
            hand_angles = [angle_between(hand_vectors[vec1_idx], hand_vectors[vec2_idx])
                           for (vec1_idx, vec2_idx) in HAND_ADJACENT_CONNECTIONS]

            angles.extend(hand_angles)

    # CUSTOM FEATURE 2: POSE DISTANCES
    if results.pose_landmarks:
        pose = [np.array([landmark.x, landmark.y, landmark.z])
                for landmark in results.pose_landmarks.landmark]

        # Extract max pose distance
        central_pose = (pose[12] + pose[11]) / 2
        dist_max_pose = max([np.linalg.norm(lm - central_pose) for lm in pose])

        # Normalize all pose landmarks
        pose = np.array([(lm - central_pose) / dist_max_pose for lm in pose])

        """
        Após os landmarks serem normalizados, escolheu-se as características a serem computa-
        das. A seguir é exibida a lista de distâncias entre os landmarks escolhidos, sejam eles do lado
        direito, do lado esquerdo ou cruzando os lados do corpo. Por exemplo, a distância entre o ombro
        e pulso pode ser entre: i) ombro esquerdo e pulso esquerdo, ii) ombro esquerdo e pulso direito,
        iii) ombro direito e pulso esquerdo, iv) ombro direito e pulso direito.
        ∙ Nariz e pulso; -> (0,16) e (0,15)
        ∙ Ombro e pulso; -> (12,16), (12,15), (11,16), (11,15)
        ∙ Ombro e cotovelo; -> (12,14), (12,13), (11,14), (11,13)
        ∙ Pulso e dedo mindinho; -> (15,17), (15,18), (16,17), (16,18)
        ∙ Pulso e dedo indicador; -> (15,19), (15,20), (16,19), (16,20)
        ∙ Dedo mindinho e dedo indicador; -> (17,19),(17,20),(18,19),(18,20)
        ∙ Dedo polegar e dedo mindinho; -> (17,21), (17,22), (18,21), (18,22)
        ∙ Dedo polegar e dedo indicador; -> (19,21), (20,21), (19,22), (20,22)
        ∙ Dedo polegar esquerdo e dedo polegar direito; -> (21,22)
        ∙ Dedo indicador esquerdo e dedo indicador direito; -> (19,20)
        ∙ Dedo mindinho esquerdo e dedo mindinho direito; -> (17,18)
        ∙ Pulso esquerdo e pulso direito; -> (15,16)
        ∙ Cotovelo esquerdo e cotovelo direito; -> (13,14)
        - nariz e cotovelo -> (0,14) e (0,13)
        """
        # Compute pose distances
        distances = [np.linalg.norm(pose[idx1] - pose[idx2])
                     for (idx1, idx2) in DESIRED_DISTANCES]

    return np.concatenate([angles, distances])


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
