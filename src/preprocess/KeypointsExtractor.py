
import cv2
import numpy as np
import pandas as pd

# TODO: REMOVE THIS FUNCTION
# Move to the appropriate module
#-----------------------------------
import mediapipe as mp

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(videos_data, show=False):
    
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils 
   
    keypoints_data = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
        print("Extracting keypoints...")
        for i in tqdm(range(len(videos_data))):
            frames_list = videos_data[i]
            frames_list_keypoints = []
 
            for frame in frames_list:
                frame, results = mediapipe_detection(frame, holistic) 
                if show:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
                    utils_cv.show(frame)

                # Mediapipe keypoints inference
                frame, results = mediapipe_detection(frame, holistic)

                # Extract handmarks
                left_hand  = np.zeros(21*3)
                right_hand = np.zeros(21*3) 
                if results.left_hand_landmarks:
                    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() 
                if results.right_hand_landmarks:
                    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() 
               
                frame_keypoints = np.concatenate([left_hand, right_hand])
                frames_list_keypoints.append(frame_keypoints)

            keypoints_data.append(frames_list_keypoints)

    return keypoints_data

#-----------------------------------

def generate_column_names(handmark_names, num_frames):
    columns = []
    for frame in range(num_frames):
        for handmark in handmark_names:
            columns.append(f"{handmark}_x_left_frame{frame}")
            columns.append(f"{handmark}_y_left_frame{frame}")
            columns.append(f"{handmark}_z_left_frame{frame}")

        for handmark in handmark_names: 
            columns.append(f"{handmark}_x_right_frame{frame}")
            columns.append(f"{handmark}_y_right_frame{frame}")
            columns.append(f"{handmark}_z_right_frame{frame}")
    columns.append("label")
    return columns


def extract_keypoints():
    handmarks_names  = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", 
                        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP",
                        "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP",
                        "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP",
                        "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]

    # Extract keypoints and reshape
    keypoints_data = np.array(extract_keypoints(videos_data, show=False)) 
    keypoints_data = keypoints_data.reshape(len(keypoints_data), -1)

    # Generate column names and append labels
    columns_names  = generate_column_names(handmarks_names, number_of_frames)
    dataset        = np.hstack((keypoints_data, np.array(labels).reshape(-1, 1)))

    # Create dataframe and save it
    df = pd.DataFrame(dataset, columns=columns_names)
    df.to_csv("data/dataset.csv", index=False)

    
