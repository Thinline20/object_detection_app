import os

import cv2


def extract_frames(video_path: str, out_dir: str) -> str:
    """Extract frames as PNG file from video and save them to folder.

    Args:
        video_path (str): Path to video file
        out_dir (str): Path to folder where frames will be saved

    Returns:
        str: Path to folder where frames were saved
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    capture = cv2.VideoCapture(video_path)

    index = 0

    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            break

        filename = os.path.join(out_dir, f"{index:06d}.png")

        if not os.path.isfile(filename):
            cv2.imwrite(filename, frame)

        index += 1

    capture.release()

    return out_dir


def load_frames(path: str) -> list:
    """Load frames from folder.

    Args:
        path (str): Path to folder

    Returns:
        list: List of frames
    """
    frames = []

    for filename in os.listdir(path):
        if filename.endswith(".png"):
            frame = cv2.imread(os.path.join(path, filename))
            frames.append(frame)

    return frames


def extract_frames_realtime(video_path: str, target_fps=1):
    """Extract frames as cv2 image object from video and save them to folder.

    Args:
        video_path (str): Path to video file
        target_fps (int, optional): Target fps. Defaults to 1

    Yield:
        frame: cv2 image object

    Usage:
        ```python
        frame_gen = extract_frames_realtime("data/traffic.mp4", 1)

        for frame in frame_gen:
            plt.imshow(frame, interpolation="bilinear")
            plt.axis("off")
            plt.show()
        ```
    """
    capture = cv2.VideoCapture(video_path)

    capture_fps = round(capture.get(cv2.CAP_PROP_FPS))
    frame_interval = int(capture_fps / target_fps)

    current_frame = 0

    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            capture.release()
            return

        if current_frame % frame_interval == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        current_frame += 1