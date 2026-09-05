# Food Calorie Estimation from Webcam

Real-time food calorie estimation using computer vision and deep learning. The system captures a live webcam feed, detects food items, estimates the food type and calorie content, and displays the results on screen.

## Overview

The project is built as a 3-module pipeline:

### 1. Food Detection

Detects and localizes food items in an image.

- **Input:** RGB image (single frame)
- **Output:** list of bounding boxes with confidence scores

```python
  detect_food(image) -> List[{"bbox": [x_min, y_min, x_max, y_max], "confidence": float}]
```

### 2. Calorie & Food-Type Estimation

Estimates the food type and calorie value from a localized food image.

- **Input:** cropped image of a single food item (from the detection module's bounding box)
- **Output:** predicted food label and estimated calorie value

```python
  estimate_calories(cropped_image) -> {"food_label": str, "calories": float}
```

### 3. Webcam Integration

Connects the webcam feed to both models and renders the live demo.

- **Input:** live webcam feed
- **Output:** real-time video display with bounding box, food label, and calorie value overlaid

## Team

- **Detection Team:** food detection model (bounding box output)
- **Calorie Team:** food type and calorie estimation model
- **Integration:** webcam capture, pipeline wiring, and live demo
