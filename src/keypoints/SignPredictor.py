import cv2
import numpy as np
import time
import torch

from ..preprocess import SampleAugment

from ..utils import utils_cv

from .KeypointsDetector import draw_landmarks
from .KeypointsDetector import extract_angles_distances_from_results
from .KeypointsDetector import extract_raw_keypoints_from_results
from .KeypointsDetector import mediapipe_detection

from .mpLoader import MediaPipeLoader

from cv2.typing import MatLike

class SignPredictor:
    def __init__(self, model_path: str) -> None:
        # Loads the model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(model_path, map_location=torch.device(self.device))
        self.model = self.model.to(self.device)
        self.model.eval() 

        # List of all supported actions
        self.ACTIONS = [
            'ABACAXI', 'ACOMPANHAR', 'ACONTECER', 'ACORDAR', 'ACRESCENTAR', 'ALTO',
            'AMIGO', 'ANO', 'ANTES', 'APAGAR', 'APRENDER', 'AR', 'BARBA', 'BARCO',
            'BICICLETA', 'BODE', 'BOI', 'BOLA', 'BOLSA', 'CABELO', 'CAIR', 'CAIXA',
            'CALCULADORA', 'CASAMENTO', 'CAVALO', 'CEBOLA', 'CERVEJA', 'CHEGAR',
            'CHINELO', 'COCO', 'COELHO', 'COMER', 'COMPARAR', 'COMPRAR', 'COMPUTADOR',
            'DESTRUIR', 'DIA', 'DIMINUIR', 'ELEFANTE', 'ELEVADOR', 'ESCOLA', 'ESCOLHER',
            'ESQUECER', 'FLAUTA', 'FLOR', 'MELANCIA', 'MISTURAR', 'NADAR', 'PATINS'
        ]

    #--------------------------------------------------------------------------------------------------------

    def predict_from_video(self, video_path: str, frame_size: tuple, mp: MediaPipeLoader) -> np.ndarray:
        rng = np.random.default_rng(seed=77796983)
        frames = SampleAugment.sample_frames(rng, video_path)
        frames = [utils_cv.resize_to(frame, frame_size) for frame in frames]

        input_size = list(self.model.parameters())[0].size()[0]
        features = []
        for frame in frames:
            _, results = mediapipe_detection(frame, mp.model)
            if input_size == 89:
                feat = extract_angles_distances_from_results(results, mp)
            else:
                feat = extract_raw_keypoints_from_results(results, mp)
            features.append(feat)

        return self.predict_from_features(features)

    # Make a model prediction for a given input
    def predict_from_features(self, input_data: list) -> np.ndarray:
        with torch.no_grad():
            input_data = torch.tensor(np.array([input_data]), dtype=torch.float32)
            outputs    = self.model(input_data.to(self.device))
            outputs    = outputs.cpu().numpy()[0]
        
        return outputs

    # Return the top k predictions from a given model output
    def get_top_i_predictions(self, outputs: np.ndarray, i: int) -> list[tuple[str, float]]:
        sorted_indices = np.argsort(outputs)[::-1]
        top_i_indices = sorted_indices[:i]
        top_i_predictions = [(self.ACTIONS[idx], outputs[idx]) for idx in top_i_indices]
        return top_i_predictions


#--------------------------------------------------------------------------------------------------------

# Turn on the webcam and run MediaPipe model inference
# on each frame, draw keypoints and predict sign
def live_sign_detection(window_size: tuple, frame_size: tuple, threshold: float, mp: MediaPipeLoader, sp: SignPredictor) -> None:
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

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
            keypoints = extract_angles_distances_from_results(results, mp)
            sequence.append(keypoints)
            sequence = sequence[-15:]

            if len(sequence) == 15:
                res  = sp.predict_from_features(sequence)
                pred = np.argmax(res)
                predictions.append(pred)
    
                # if the current prediction is the same as the last 10 frames
                if np.unique(predictions[-10:])[0] == pred:
                    # if the current prediction has a high enough probability
                    if res[pred] > threshold:
                        # if the current sign is different than the last sign
                        if len(sentence) == 0 or sp.ACTIONS[pred] != sentence[-1]:
                            sentence.append(sp.ACTIONS[pred])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Show the 5 most likely actions for this sequence
                top_5 = sp.get_top_i_predictions(res, 5)
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
                if make_predictions:
                    time.sleep(3)
                    print("INICIANDO DETECÇÃO...")
        
    vidcap.release()
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------------

# Write the predictions on the image
def draw_predictions(image: MatLike, predictions: np.ndarray) -> MatLike:
    for idx, (pred, conf) in enumerate(predictions):
        cv2.putText(image, f"{conf:.3f} {pred}", (0, 85+idx*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return image

#--------------------------------------------------------------------------------------------------------
