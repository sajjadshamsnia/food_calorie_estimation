from src.calorie_detection import get_calories
from src.food_detector import FoodDetector

import tkinter as tk
import cv2
import time

from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "./models/yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.50

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# ============================================================
# TEMPORARY DEVELOPMENT DATA
# ============================================================
#
# DO NOT build the final system around this.
#
# This is only being used until the other groups provide:
#   1. The real food detection model
#   2. The real food/portion/calorie data
#
# Later, the integration person can replace this entire
# development lookup with the real pipeline.
# ============================================================

TEMP_FOOD_DATA = {
    "apple": {
        "calories": 95,
        "portion": "1 medium"
    },

    "banana": {
        "calories": 105,
        "portion": "1 medium"
    },

    "orange": {
        "calories": 62,
        "portion": "1 medium"
    },

    "pizza": {
        "calories": 285,
        "portion": "1 slice"
    },

    "sandwich": {
        "calories": 300,
        "portion": "1 sandwich"
    },

    "cake": {
        "calories": 350,
        "portion": "1 slice"
    },

    "hot dog": {
        "calories": 250,
        "portion": "1 hot dog"
    },

    "donut": {
        "calories": 190,
        "portion": "1 donut"
    },

    "broccoli": {
        "calories": 55,
        "portion": "100 g"
    },

    "carrot": {
        "calories": 25,
        "portion": "1 medium"
    },
}


# ============================================================
# COLORS
# ============================================================

BG = "#0F1117"
PANEL = "#171A23"
PANEL_LIGHT = "#1E222D"

ACCENT = "#8B5CF6"
ACCENT_HOVER = "#9F75F8"

TEXT = "#F5F5F7"
TEXT_SECONDARY = "#A7AAB5"
TEXT_MUTED = "#6F7380"

SUCCESS = "#4ADE80"
DANGER = "#F87171"

WHITE = "#FFFFFF"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    model = None
    print("Could not load YOLO model:")
    print(e)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


# ============================================================
# STATE
# ============================================================

camera_running = True
scanning = True

last_detections = []

previous_time = time.time()
fps = 0


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("NutriScan")
root.geometry("1280x820")
root.minsize(1050, 700)
root.configure(bg=BG)


# ============================================================
# HELPER
# ============================================================

def create_label(
    parent,
    text,
    font,
    color=TEXT,
    **kwargs
):
    return tk.Label(
        parent,
        text=text,
        font=font,
        fg=color,
        bg=parent.cget("bg"),
        **kwargs
    )


def rounded_button(
    parent,
    text,
    command,
    width=14
):
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=("Segoe UI", 10, "bold"),
        fg=WHITE,
        bg=ACCENT,
        activeforeground=WHITE,
        activebackground=ACCENT_HOVER,
        relief="flat",
        bd=0,
        cursor="hand2",
        width=width,
        padx=10,
        pady=8
    )


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=BG
)

header.pack(
    fill="x",
    padx=30,
    pady=(25, 15)
)


# Logo

logo = tk.Label(
    header,
    text="N",
    font=("Segoe UI", 22, "bold"),
    fg=WHITE,
    bg=ACCENT,
    width=2,
    height=1
)

logo.pack(side="left")


# Title

title_frame = tk.Frame(
    header,
    bg=BG
)

title_frame.pack(
    side="left",
    padx=14
)


title = create_label(
    title_frame,
    "NutriScan",
    ("Segoe UI", 22, "bold")
)

title.pack(anchor="w")


subtitle = create_label(
    title_frame,
    "AI-powered food & calorie estimation",
    ("Segoe UI", 10),
    TEXT_SECONDARY
)

subtitle.pack(anchor="w")


# Camera status

status_frame = tk.Frame(
    header,
    bg=PANEL
)

status_frame.pack(
    side="right"
)


status_dot = tk.Label(
    status_frame,
    text="●",
    font=("Segoe UI", 12),
    fg=SUCCESS,
    bg=PANEL
)

status_dot.pack(
    side="left",
    padx=(12, 5),
    pady=8
)


status_text = tk.Label(
    status_frame,
    text="Camera connected",
    font=("Segoe UI", 9, "bold"),
    fg=TEXT,
    bg=PANEL
)

status_text.pack(
    side="left",
    padx=(0, 12),
    pady=8
)


# ============================================================
# MAIN CONTENT
# ============================================================

content = tk.Frame(
    root,
    bg=BG
)

content.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=10
)

content.grid_columnconfigure(
    0,
    weight=3
)

content.grid_columnconfigure(
    1,
    weight=1
)

content.grid_rowconfigure(
    0,
    weight=1
)


# ============================================================
# CAMERA PANEL
# ============================================================

camera_panel = tk.Frame(
    content,
    bg=PANEL,
    highlightthickness=1,
    highlightbackground="#252936"
)

camera_panel.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 15)
)

camera_panel.grid_rowconfigure(
    1,
    weight=1
)

camera_panel.grid_columnconfigure(
    0,
    weight=1
)


# Camera header

camera_header = tk.Frame(
    camera_panel,
    bg=PANEL
)

camera_header.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=20,
    pady=15
)


camera_title = create_label(
    camera_header,
    "Live Camera",
    ("Segoe UI", 14, "bold")
)

camera_title.pack(
    side="left"
)


fps_label = tk.Label(
    camera_header,
    text="FPS: --",
    font=("Segoe UI", 9),
    fg=TEXT_SECONDARY,
    bg=PANEL
)

fps_label.pack(
    side="right"
)


# Camera display

camera_display = tk.Label(
    camera_panel,
    bg="#080A0F",
    text="Starting camera...",
    fg=TEXT_MUTED,
    font=("Segoe UI", 12)
)

camera_display.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=20,
    pady=(0, 15)
)


# Controls

controls = tk.Frame(
    camera_panel,
    bg=PANEL
)

controls.grid(
    row=2,
    column=0,
    sticky="ew",
    padx=20,
    pady=(0, 20)
)


# ============================================================
# RESULTS PANEL
# ============================================================

results_panel = tk.Frame(
    content,
    bg=PANEL,
    highlightthickness=1,
    highlightbackground="#252936"
)

results_panel.grid(
    row=0,
    column=1,
    sticky="nsew"
)


# Header

results_header = tk.Frame(
    results_panel,
    bg=PANEL
)

results_header.pack(
    fill="x",
    padx=20,
    pady=(20, 10)
)


results_title = create_label(
    results_header,
    "Detection Results",
    ("Segoe UI", 14, "bold")
)

results_title.pack(
    anchor="w"
)


results_subtitle = create_label(
    results_header,
    "Foods detected in the current frame",
    ("Segoe UI", 9),
    TEXT_SECONDARY
)

results_subtitle.pack(
    anchor="w",
    pady=(3, 0)
)


# ============================================================
# RESULTS SCROLL AREA
# ============================================================

results_area = tk.Frame(
    results_panel,
    bg=PANEL
)

results_area.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=10
)


# ============================================================
# RESULT FUNCTIONS
# ============================================================

def clear_results():

    for widget in results_area.winfo_children():
        widget.destroy()


def display_empty_results():

    clear_results()

    empty = tk.Frame(
        results_area,
        bg=PANEL
    )

    empty.pack(
        expand=True,
        fill="both"
    )

    icon = tk.Label(
        empty,
        text="◌",
        font=("Segoe UI", 38),
        fg=TEXT_MUTED,
        bg=PANEL
    )

    icon.pack(
        pady=(50, 10)
    )

    text = tk.Label(
        empty,
        text="No food detected",
        font=("Segoe UI", 11, "bold"),
        fg=TEXT_SECONDARY,
        bg=PANEL
    )

    text.pack()

    hint = tk.Label(
        empty,
        text="Point the camera at food",
        font=("Segoe UI", 9),
        fg=TEXT_MUTED,
        bg=PANEL
    )

    hint.pack(
        pady=5
    )


def create_food_card(detection):

    card = tk.Frame(
        results_area,
        bg=PANEL_LIGHT,
        highlightthickness=1,
        highlightbackground="#292D38"
    )

    card.pack(
        fill="x",
        padx=5,
        pady=6
    )

    name = detection["name"]
    confidence = detection["confidence"]

    calories = detection.get("calories")
    portion = detection.get("portion")

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name_label = tk.Label(
        card,
        text=name.title(),
        font=("Segoe UI", 11, "bold"),
        fg=TEXT,
        bg=PANEL_LIGHT
    )

    name_label.pack(
        anchor="w",
        padx=12,
        pady=(10, 2)
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence_label = tk.Label(
        card,
        text=f"Confidence: {confidence * 100:.0f}%",
        font=("Segoe UI", 8),
        fg=TEXT_SECONDARY,
        bg=PANEL_LIGHT
    )

    confidence_label.pack(
        anchor="w",
        padx=12
    )

    # --------------------------------------------------------
    # CALORIES
    # --------------------------------------------------------

    if calories is not None:

        calorie_text = f"{calories} kcal"
        calorie_color = ACCENT

    else:

        calorie_text = "Calorie estimate pending"
        calorie_color = TEXT_MUTED

    calorie_label = tk.Label(
        card,
        text=calorie_text,
        font=("Segoe UI", 12, "bold"),
        fg=calorie_color,
        bg=PANEL_LIGHT
    )

    calorie_label.pack(
        anchor="w",
        padx=12,
        pady=(8, 0)
    )

    # --------------------------------------------------------
    # PORTION
    # --------------------------------------------------------

    if portion:

        portion_label = tk.Label(
            card,
            text=f"Portion: {portion}",
            font=("Segoe UI", 8),
            fg=TEXT_SECONDARY,
            bg=PANEL_LIGHT
        )

        portion_label.pack(
            anchor="w",
            padx=12,
            pady=(2, 10)
        )

    else:

        spacer = tk.Frame(
            card,
            height=8,
            bg=PANEL_LIGHT
        )

        spacer.pack()


def display_detections(detections):

    clear_results()

    if not detections:

        display_empty_results()
        return

    total_calories = 0
    calorie_count = 0

    # --------------------------------------------------------
    # EVERY DETECTION GETS ITS OWN CARD
    # --------------------------------------------------------

    for detection in detections:

        create_food_card(
            detection
        )

        calories = detection.get(
            "calories"
        )

        if calories is not None:

            total_calories += calories
            calorie_count += 1

    # --------------------------------------------------------
    # TOTAL CARD
    # --------------------------------------------------------
    #
    # Kept at the bottom of the result list.
    #
    # This is deliberately calculated from the detections,
    # rather than from a separate hardcoded list.
    # --------------------------------------------------------

    total_card = tk.Frame(
        results_area,
        bg=ACCENT
    )

    total_card.pack(
        fill="x",
        padx=5,
        pady=(15, 5)
    )

    total_title = tk.Label(
        total_card,
        text="TOTAL ESTIMATE",
        font=("Segoe UI", 8, "bold"),
        fg="#EDE9FE",
        bg=ACCENT
    )

    total_title.pack(
        anchor="w",
        padx=14,
        pady=(12, 0)
    )

    if calorie_count > 0:

        total_text = f"{total_calories} kcal"

    else:

        total_text = "Pending"

    total_value = tk.Label(
        total_card,
        text=total_text,
        font=("Segoe UI", 20, "bold"),
        fg=WHITE,
        bg=ACCENT
    )

    total_value.pack(
        anchor="w",
        padx=14,
        pady=(2, 12)
    )


# ============================================================
# DETECTION / INTEGRATION INTERFACE
# ============================================================

def process_frame(frame):
    """
    Main processing interface.

    This function is intentionally kept separate from the UI.

    CURRENT:
        Webcam frame
            ↓
        YOLOv8
            ↓
        temporary food lookup
            ↓
        detection dictionaries

    FINAL:
        Webcam frame
            ↓
        Group 1 detector
            ↓
        integration/database
            ↓
        Group 2 calorie estimator
            ↓
        detection dictionaries
            ↓
        UI

    The UI does not need to change when that happens.
    """

    return detect_food(frame)


def detect_food(frame):
    """
    Temporary detector using YOLOv8.

    Returns MULTIPLE detections.

    Example:

    [
        {
            "name": "banana",
            "confidence": 0.91,
            "bbox": (100, 120, 300, 400),
            "calories": 105,
            "portion": "1 medium"
        },

        {
            "name": "apple",
            "confidence": 0.87,
            "bbox": (400, 100, 550, 300),
            "calories": 95,
            "portion": "1 medium"
        }
    ]

    Later this function/process_frame() is where the real
    Group 1 + Group 2 + database pipeline can be connected.
    """

    if model is None:

        return []

    detections = []

    try:

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        if result.boxes is None:

            return []

        # ====================================================
        # MULTI-DETECTION
        # ====================================================

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            food_name = model.names[
                class_id
            ].lower()

            # -----------------------------------------------
            # TEMPORARY FOOD FILTER
            # -----------------------------------------------
            #
            # yolov8n.pt is trained on COCO.
            #
            # Therefore it can detect people, chairs,
            # bottles, cars, etc.
            #
            # Only known food classes are accepted for now.
            #
            # Group 1's food-specific model will replace this.
            # -----------------------------------------------

            if food_name not in TEMP_FOOD_DATA:

                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            food_data = TEMP_FOOD_DATA[
                food_name
            ]

            detection = {

                "name": food_name,

                "confidence": confidence,

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                # TEMPORARY
                "calories": food_data[
                    "calories"
                ],

                # TEMPORARY
                "portion": food_data[
                    "portion"
                ]
            }

            detections.append(
                detection
            )

    except Exception as e:

        print(
            "Detection error:",
            e
        )

    return detections


# ============================================================
# DRAW ALL DETECTIONS
# ============================================================

def draw_detections(
    frame,
    detections
):

    # --------------------------------------------------------
    # EVERY FOOD GETS ITS OWN BOX
    # --------------------------------------------------------

    for detection in detections:

        name = detection[
            "name"
        ]

        confidence = detection[
            "confidence"
        ]

        x1, y1, x2, y2 = detection[
            "bbox"
        ]

        # Bounding box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (139, 92, 246),
            2
        )

        # Label

        label = (
            f"{name.title()} "
            f"{confidence * 100:.0f}%"
        )

        calories = detection.get(
            "calories"
        )

        if calories is not None:

            label += (
                f" | {calories} kcal"
            )

        font = cv2.FONT_HERSHEY_SIMPLEX

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            font,
            0.55,
            2
        )

        # Prevent label from going outside
        # the top of the frame.

        label_y = max(
            y1,
            text_height + 12
        )

        cv2.rectangle(
            frame,
            (
                x1,
                label_y - text_height - 12
            ),
            (
                x1 + text_width + 10,
                label_y
            ),
            (139, 92, 246),
            -1
        )

        cv2.putText(
            frame,
            label,
            (
                x1 + 5,
                label_y - 6
            ),
            font,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


# ============================================================
# SCANNING
# ============================================================

def toggle_scanning():

    global scanning

    scanning = not scanning

    if scanning:

        scan_button.config(
            text="⏸  Pause Scan"
        )

    else:

        scan_button.config(
            text="▶  Resume Scan"
        )


scan_button = rounded_button(
    controls,
    "⏸  Pause Scan",
    toggle_scanning
)

scan_button.pack(
    side="left"
)


# ============================================================
# CAMERA TOGGLE
# ============================================================

def toggle_camera():

    global camera_running

    camera_running = not camera_running

    if camera_running:

        camera_button.config(
            text="Turn Camera Off"
        )

        status_dot.config(
            fg=SUCCESS
        )

        status_text.config(
            text="Camera connected"
        )

    else:

        camera_button.config(
            text="Turn Camera On"
        )

        status_dot.config(
            fg=DANGER
        )

        status_text.config(
            text="Camera paused"
        )


camera_button = rounded_button(
    controls,
    "Turn Camera Off",
    toggle_camera
)

camera_button.pack(
    side="left",
    padx=10
)


# ============================================================
# CAMERA LOOP
# ============================================================

def update_camera():

    global previous_time
    global fps
    global last_detections

    if camera_running and camera.isOpened():

        success, frame = camera.read()

        if success:

            # Mirror webcam

            frame = cv2.flip(
                frame,
                1
            )

            # -----------------------------------------------
            # PROCESS FRAME
            # -----------------------------------------------

            if scanning:

                detections = process_frame(
                    frame
                )

                last_detections = detections

                # Draw ALL detected foods

                draw_detections(
                    frame,
                    detections
                )

                # Update sidebar

                display_detections(
                    detections
                )

            else:

                # Keep previous detections visible

                draw_detections(
                    frame,
                    last_detections
                )

            # -----------------------------------------------
            # FPS
            # -----------------------------------------------

            current_time = time.time()

            delta = (
                current_time -
                previous_time
            )

            if delta > 0:

                current_fps = 1 / delta

                if fps == 0:

                    fps = current_fps

                else:

                    fps = (
                        0.9 * fps +
                        0.1 * current_fps
                    )

            previous_time = current_time

            fps_label.config(
                text=f"FPS: {fps:.1f}"
            )

            # -----------------------------------------------
            # OPENCV → PIL
            # -----------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                frame_rgb
            )

            display_width = (
                camera_display.winfo_width()
            )

            display_height = (
                camera_display.winfo_height()
            )

            if (
                display_width > 10
                and display_height > 10
            ):

                image.thumbnail(
                    (
                        display_width,
                        display_height
                    ),
                    Image.Resampling.LANCZOS
                )

            photo = ImageTk.PhotoImage(
                image=image
            )

            camera_display.configure(
                image=photo,
                text=""
            )

            camera_display.image = photo

    root.after(
        30,
        update_camera
    )


# ============================================================
# CLEAN EXIT
# ============================================================

def close_application():

    global camera_running

    camera_running = False

    if camera.isOpened():

        camera.release()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    close_application
)


# ============================================================
# CAMERA ERROR STATE
# ============================================================

if not camera.isOpened():

    status_dot.config(
        fg=DANGER
    )

    status_text.config(
        text="Camera unavailable"
    )

    camera_display.config(
        text="Could not open webcam."
    )


# ============================================================
# START APPLICATION
# ============================================================

display_empty_results()

update_camera()

root.mainloop()