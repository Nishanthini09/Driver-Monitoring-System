import cv2
import numpy as np
from phone_detector import PhoneDetector
from drowsiness_detector import DrowsinessDetector  # <-- using your class

# --- Utils ---
def draw_panel(frame, title, lines, top_left=(10, 10), color=(0, 0, 0), bgcolor=(255, 255, 255), alpha=0.6):
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 25
    padding = 10

    width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines + [title]]) + 2 * padding
    height = line_height * (len(lines) + 1) + 2 * padding

    x, y = top_left
    cv2.rectangle(overlay, (x, y), (x + width, y + height), bgcolor, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, title, (x + padding, y + line_height), font, font_scale, color, 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + padding, y + (i + 2) * line_height), font, font_scale, color, thickness)

# --- Initialize ---
drowsiness_detector = DrowsinessDetector(ear_thresh=0.22, consec_frames=27)
phone_detector = PhoneDetector(model_path='yolov5s.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # --- Drowsiness Detection ---
    frame, ear, is_drowsy, mesh_points = drowsiness_detector.analyze(frame)

    # --- Head Pose Estimation ---
    head_direction = "Forward"
    if mesh_points is not None:
        nose_tip = mesh_points[1]
        if nose_tip[0] < w // 3:
            head_direction = "Left"
        elif nose_tip[0] > w * 2 // 3:
            head_direction = "Right"

    # --- Phone Detection ---
    phone_in_use = phone_detector.detect(frame)

    # --- Panels ---
    head_lines = [f"Direction: {head_direction}"]
    if head_direction != "Forward":
        head_lines.append("!! Not facing front!")

    drowsy_lines = [
        f"EAR: {ear:.2f}" if ear is not None else "EAR: N/A",
        f"PERCLOS: {round(1 - ear, 2) if ear is not None else 'N/A'}"
    ]
    if is_drowsy:
        drowsy_lines.append("!! DROWSINESS DETECTED!")

    phone_lines = [f"Phone: {'DETECTED' if phone_in_use else 'Not in use'}"]

    draw_panel(frame, "HEAD POSE", head_lines, top_left=(10, 10))
    draw_panel(frame, "DROWSINESS", drowsy_lines, top_left=(10, 150))
    draw_panel(frame, "PHONE USAGE", phone_lines, top_left=(10, 290))

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
