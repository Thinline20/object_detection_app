import cv2
from matplotlib import pyplot as plt
from ultralyticsplus import YOLO, render_result
import streamlit as st
import glob
import queue
from collections import Counter
import threading
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd

model = YOLO("ultralyticsplus/yolov8n")

# set model parameters
model.overrides["conf"] = 0.25  # NMS confidence threshold
model.overrides["iou"] = 0.45  # NMS IoU threshold
model.overrides["agnostic_nms"] = False  # NMS class-agnostic
model.overrides["max_det"] = 1000  # maximum number of detections per image

base_label = {
    0: "__background__",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
}


dangerous_level = {
    "person": 1,
    "bicycle": 5,
    "car": 5,
    "motorcycle": 5,
    "airplane": 5,
    "bus": 5,
    "train": 5,
    "truck": 5,
    "boat": 5,
    "traffic light": 4,
    "fire hydrant": 3,
    "stop sign": 2,
    "parking meter": 2,
    "bench": 2,
    "bird": 3,
    "cat": 3,
    "dog": 3,
    "horse": 3,
    "sheep": 3,
    "cow": 5,
    "elephant": 5,
    "bear": 5,
    "zebra": 5,
    "giraffe": 5,
    "backpack": 1,
    "umbrella": 1,
    "handbag": 1,
    "tie": 1,
    "suitcase": 1,
    "frisbee": 3,
    "skis": 3,
    "snowboard": 3,
    "sports ball": 3,
    "kite": 1,
    "baseball bat": 3,
    "baseball glove": 3,
    "skateboard": 3,
    "tennis racket": 3,
    "bottle": 1,
    "wine glass": 3,
    "cup": 3,
    "fork": 4,
    "knife": 5,
    "spoon": 1,
    "bowl": 3,
    "banana": 1,
    "apple": 1,
    "sandwich": 1,
    "orange": 1,
    "brocoli": 1,
    "carrot": 1,
    "hot dog": 1,
    "pizza": 1,
    "donut": 1,
    "cake": 1,
    "chair": 3,
    "couch": 1,
    "potted plant": 4,
    "bed": 1,
    "dining table": 1,
    "toilet": 1,
    "tv": 1,
    "laptop": 1,
    "mouse": 1,
    "remote": 1,
    "keyboard": 1,
    "cell phone": 1,
    "microwave": 1,
    "oven": 5,
    "toaster": 4,
    "sink": 1,
    "refrigerator": 1,
    "book": 1,
    "clock": 1,
    "vase": 1,
    "scissors": 1,
    "teddy bear": 1,
    "hair drier": 3,
    "toothbrush": 1,
}      


def box_label(image, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_bboxes(image, boxes, score=True, conf=None):
    # Define COCO Labels
    labels = base_label
    # Define colors
    # colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
    colors = [
        (89, 161, 197),
        (67, 161, 255),
        (19, 222, 24),
        (186, 55, 2),
        (167, 146, 11),
        (190, 76, 98),
        (130, 172, 179),
        (115, 209, 128),
        (204, 79, 135),
        (136, 126, 185),
        (209, 213, 45),
        (44, 52, 10),
        (101, 158, 121),
        (179, 124, 12),
        (25, 33, 189),
        (45, 115, 11),
        (73, 197, 184),
        (62, 225, 221),
        (32, 46, 52),
        (20, 165, 16),
        (54, 15, 57),
        (12, 150, 9),
        (10, 46, 99),
        (94, 89, 46),
        (48, 37, 106),
        (42, 10, 96),
        (7, 164, 128),
        (98, 213, 120),
        (40, 5, 219),
        (54, 25, 150),
        (251, 74, 172),
        (0, 236, 196),
        (21, 104, 190),
        (226, 74, 232),
        (120, 67, 25),
        (191, 106, 197),
        (8, 15, 134),
        (21, 2, 1),
        (142, 63, 109),
        (133, 148, 146),
        (187, 77, 253),
        (155, 22, 122),
        (218, 130, 77),
        (164, 102, 79),
        (43, 152, 125),
        (185, 124, 151),
        (95, 159, 238),
        (128, 89, 85),
        (228, 6, 60),
        (6, 41, 210),
        (11, 1, 133),
        (30, 96, 58),
        (230, 136, 109),
        (126, 45, 174),
        (164, 63, 165),
        (32, 111, 29),
        (232, 40, 70),
        (55, 31, 198),
        (148, 211, 129),
        (10, 186, 211),
        (181, 201, 94),
        (55, 35, 92),
        (129, 140, 233),
        (70, 250, 116),
        (61, 209, 152),
        (216, 21, 138),
        (100, 0, 176),
        (3, 42, 70),
        (151, 13, 44),
        (216, 102, 88),
        (125, 216, 93),
        (171, 236, 47),
        (253, 127, 103),
        (205, 137, 244),
        (193, 137, 224),
        (36, 152, 214),
        (17, 50, 238),
        (154, 165, 67),
        (114, 129, 60),
        (119, 24, 48),
        (73, 8, 110),
    ]

    # plot each boxes
    current_labels = []

    for box in boxes:
        # add score in label if score=True
        if score:
            label = (
                labels[int(box[-1]) + 1]
                + " "
                + str(round(100 * float(box[-2]), 1))
                + "%"
            )
        else:
            label = labels[int(box[-1]) + 1]

        current_labels.append(label)

        # filter every box under conf threshold if conf threshold setted
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)

    # show image
    return image, current_labels


fin = False


def process_video(video, output_path: str, conf: float, iou: float, agnostic_nms: bool):
    if output_path == "":
        output_path = "out/output.mp4"
    global fin

    fin = False

    model.overrides["conf"] = conf
    model.overrides["iou"] = iou
    model.overrides["agnostic_nms"] = agnostic_nms

    capture = cv2.VideoCapture(video)

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"x264"), fps, (width, height)
    )

    ret, frame = capture.read()

    while ret:
        detection = model(frame)

        processed_frame, labels = plot_bboxes(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            detection[0].boxes.boxes,
            score=False,
        )

        video.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        ret, frame = capture.read()

        q.put([processed_frame, labels, None])

    fin = True
    capture.release()

    q.put([processed_frame, labels, True])
    
    return


is_exit_target_if_main_exits = True


# StreamLit application
st.title("Object Detection")

video_files = glob.glob("data/*.mp4")
video_files.insert(0, "")

stop_process = False

q = queue.Queue()

data = [None, None, None]

with st.container():
    if len(video_files) == 0:
        st.write(
            f"Put some video files in directory `data/` to activate this player."
        )

    else:
        filename = st.selectbox(
            "Select a video file from directory `data/`", video_files, 0
        )
        if filename != "":
            video = st.video(filename)

    conf_slider = st.slider("Confidence", 0.0, 1.0, 0.5, 0.01)
    iou_slider = st.slider("IoU", 0.0, 1.0, 0.45, 0.01)
    agnostic_nms_checkbox = st.checkbox("Agnostic NMS")
    output_path = st.text_input("Output path", "out/output.mp4")

    run = st.button("Run")

    if run:
        processed_video = None

if run:
    frame1 = st.empty()
    frame2 = st.empty()
    frame3 = st.empty()

    threading.Thread(
        target=process_video,
        daemon=is_exit_target_if_main_exits,
        args=(
            filename,
            output_path,
            conf_slider,
            iou_slider,
            agnostic_nms_checkbox,
        ),
    ).start()
    
    while True:
        data = q.get()

        danger_list = {i: [0, 0] for i in set(data[1])}
        counter = dict(Counter(data[1]))
        hist_data = [0, 0, 0, 0, 0]
        
        for i, j in danger_list.items():
            j[0] = dangerous_level[i]
            j[1] = counter[i]
            
            hist_data[j[0] - 1] += j[1]

        # df = pd.DataFrame.from_dict(counter, orient="index").reset_index()
        df = pd.DataFrame.from_dict(danger_list, orient="index", columns=["level", "count"])

        with st.container():
            with frame1.container():
                st.image(data[0], channels="RGB")
                
            with frame2.container():
                fig = px.bar(x=range(1,6), y=hist_data)

                st.plotly_chart(fig)
                st.dataframe(df, use_container_width=True)
                
            if data[2]:
                break

    with frame3.container():
        st.video(output_path)
        st.markdown(output_path)