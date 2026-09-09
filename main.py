import tkinter as tk
import cv2
import time

from PIL import Image, ImageTk
from src.food_detector import FoodDetector
from src.calorie_detection import get_calories


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "./models/yolov8s_food41_enriched_v1.pt"

CONFIDENCE_THRESHOLD = 0.50

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


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
# LOAD TEAM A DETECTOR
# ============================================================

try:
    food_detector = FoodDetector(
        model_path=MODEL_PATH,
        conf=0.25,
        primary_imgsz=640,
        fallback_imgsz=1280
    )
except Exception as e:
    food_detector = None
    print("Could not load food detector:")
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

    kcal_per_gram = detection.get("kcal_per_gram")
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

    if kcal_per_gram is not None:

        calorie_text = f"{kcal_per_gram:.3f} kcal/g"
        calorie_color = ACCENT

    else:

        calorie_text = "Energy density unavailable"
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

    for detection in detections:
        create_food_card(detection)

    # The Team B JSON contains kcal per gram, not total calories.
    # Therefore values must not be summed as a total-calorie estimate.
    summary_card = tk.Frame(
        results_area,
        bg=ACCENT
    )

    summary_card.pack(
        fill="x",
        padx=5,
        pady=(15, 5)
    )

    summary_title = tk.Label(
        summary_card,
        text="DETECTED FOODS",
        font=("Segoe UI", 8, "bold"),
        fg="#EDE9FE",
        bg=ACCENT
    )
    summary_title.pack(anchor="w", padx=14, pady=(12, 0))

    summary_value = tk.Label(
        summary_card,
        text=str(len(detections)),
        font=("Segoe UI", 20, "bold"),
        fg=WHITE,
        bg=ACCENT
    )
    summary_value.pack(anchor="w", padx=14, pady=(2, 12))


# ============================================================
# DETECTION / INTEGRATION INTERFACE
# ============================================================

def process_frame(frame):
    """
    Team C integration interface.

    Input:
        OpenCV webcam frame in BGR format.

    Pipeline:
        BGR webcam frame
            -> RGB conversion
            -> Team A FoodDetector
            -> class name / confidence / bbox
            -> Team B nutrition lookup (kcal per gram)
            -> UI-compatible detection dictionaries
    """

    if food_detector is None:
        return []

    try:
        # Team A wrapper expects an RGB NumPy image.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        team_a_results = food_detector.predict(frame_rgb)

        detections = []

        for item in team_a_results:
            food_name = str(item["class_name"]).strip().lower()
            kcal_per_gram = get_calories(food_name)

            # Ignore a detection if the nutrition lookup has no matching class.
            if kcal_per_gram is None:
                continue

            detections.append({
                "name": food_name,
                "confidence": float(item["confidence"]),
                "bbox": tuple(item["bbox"]),
                "kcal_per_gram": float(kcal_per_gram),
                "portion": None
            })

        return detections

    except Exception as e:
        print("Pipeline error:", e)
        return []


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

        kcal_per_gram = detection.get(
            "kcal_per_gram"
        )

        if kcal_per_gram is not None:

            label += (
                f" | {kcal_per_gram:.3f} kcal/g"
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