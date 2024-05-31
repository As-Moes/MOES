
import mediapipe as mp

class MediaPipeLoader:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing  = mp.solutions.drawing_utils 
        self.model       = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        """
        More information about MediaPipe keypoints:
        https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
        """ 
        self.HAND_ADJACENT_CONNECTIONS = [
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
      
        """
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
        ∙ nariz e cotovelo -> (0,14) e (0,13)        
        """
        self.TARGET_DISTANCES = [
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
