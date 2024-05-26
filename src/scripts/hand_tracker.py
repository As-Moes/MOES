import cv2
import mediapipe as mp

# Initialize mediapipe hands module
mphands   = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def track_hands(winwidth: int, winheight: int) -> None:
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    with mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, hands)

            # Draw landmarks on the frame
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mpdrawing.draw_landmarks(
                        image, lm, mphands.HAND_CONNECTIONS)

            image = cv2.resize(image, (winwidth, winheight))

            cv2.imshow('Hand Tracking', cv2.flip(image, 1))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_hands(1080, 900)
