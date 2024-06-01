import os

import cv2


def cut_problematic_frame(videos: list) -> None:
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        new_path, ext = os.path.splitext(video_path)
        new_path += "-edited" + ext
        output = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for i in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                output.write(frame)

        cap.release()
        output.release()
