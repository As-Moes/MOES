
import cv2
import numpy as np
import torch

from ..utils import utils_cv
from ..utils import utils_math

from ..train import Trainer

from .mpLoader import ACTIONS
from .mpLoader import MediaPipeLoader

from cv2.typing import MatLike
from typing import NamedTuple

#--------------------------------------------------------------------------------------------------------

# Run MediaPipe model inference
def mediapipe_detection(image: MatLike, model) -> tuple[MatLike, NamedTuple]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
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
    
    return __extract_keypoints_raw(results)
    
def __extract_keypoints_raw(results: NamedTuple) -> np.ndarray:
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
    
    return __extract_angles_distances(results, mp)
    
def __extract_angles_distances(results: NamedTuple, mp: MediaPipeLoader) -> np.ndarray:
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

    show_feed = True
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret: break
    
        image, results = mediapipe_detection(frame, mp.model)
        image          = draw_landmarks(image if show_feed else np.zeros(image.shape, dtype=np.uint8), results, mp)
        image          = utils_cv.resize_to(image, window_size)
        
        cv2.imshow('Live', image)
        match chr(cv2.waitKey(1) & 0xFF):
            case 'q':
                break
            case 't':
                show_feed = not show_feed
        
    vidcap.release()
    cv2.destroyAllWindows()

# Turn on the webcam and run MediaPipe model inference
# on each frame, draw keypoints and predict sign
def live_sign_detection(window_size: tuple, frame_size: tuple, threshold: float, mp: MediaPipeLoader, model_path: str, train_dataset_path, val_dataset_path, test_dataset_path) -> None:
    # Loads the model
    batch_size    = 1
    _, input_size, num_classes = Trainer.load_dataset(train_dataset_path, val_dataset_path, test_dataset_path, batch_size) 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Trainer.create_lstm_amanda(input_size, num_classes, device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device) 
    model.eval() 

    # Sets up video capture
    vidcap = cv2.VideoCapture(0)
    # vidcap = cv2.VideoCapture("data/Alto.mp4")
    vidcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # Main loop
    show_feed        = True   # press 't' to toggle background
    show_keypoints   = True   # press 'k' to toggle keypoints
    make_predictions = False  # press 'p' to start predicting
    sequence    = []
    predictions = []
    sentence    = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break
        # Resize the frame to the size used in training
        frame = utils_cv.resize_to(frame, frame_size)

        image, results = mediapipe_detection(frame, mp.model)

        if not show_feed:  # replace image with a black background
            image = np.zeros(image.shape, dtype=np.uint8)

        if show_keypoints:
            image = draw_landmarks(image, results, mp)
        image = utils_cv.resize_to(image, window_size)
    
        if make_predictions:
            keypoints = __extract_angles_distances(results, mp)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res  = predict(sequence, model, device)
                pred = np.argmax(res)
                predictions.append(pred)
    
                # if the current prediction is the same as the last 10 frames
                if np.unique(predictions[-10:])[0] == pred:
                    # if the current prediction has a high enough probability
                    if res[pred] > threshold:
                        # if the current sign is different than the last sign
                        if len(sentence) == 0 or ACTIONS[pred] != sentence[-1]:
                            sentence.append(ACTIONS[pred])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Show the 5 most likely actions for this sequence
                top_5 = get_top_i_predictions(res, 5)
                image = draw_predictions(image, top_5)

            # Show the last 5 (distinct) predictions on screen
            cv2.rectangle(image, (0,0), (window_size[0], 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Live', image)
        match chr(cv2.waitKey(1) & 0xFF):
            case 'q':
                break
            case 't':
                show_feed = not show_feed
            case 'k':
                show_keypoints = not show_keypoints
            case 'p':
                make_predictions = not make_predictions
        
    vidcap.release()
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------------

# Make a model prediction for a given input
def predict(input_data: list, model: torch.nn.Module, device: str) -> np.ndarray:
    with torch.no_grad():
        input_data = torch.tensor(np.array([input_data]), dtype=torch.float32)
        outputs    = model(input_data.to(device))
        outputs    = outputs.cpu().numpy()[0]
    
    return outputs

# Return the top k predictions from a given model output
def get_top_i_predictions(outputs: np.ndarray, i: int) -> list[tuple[str, float]]:
    sorted_indices = np.argsort(outputs)[::-1]
    top_i_indices = sorted_indices[:i]
    top_i_predictions = [(ACTIONS[idx], outputs[idx]) for idx in top_i_indices]
    return top_i_predictions

# Write the predictions on the image
def draw_predictions(image: MatLike, predictions: np.ndarray) -> MatLike:
    for idx, (pred, conf) in enumerate(predictions):
        cv2.putText(image, f"{conf:.3f} {pred}", (0, 85+idx*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return image

#--------------------------------------------------------------------------------------------------------
