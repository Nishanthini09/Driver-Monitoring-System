import cv2
import numpy as np
import mediapipe as mp
from phone_detector import PhoneDetector

# --- Constants ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
EAR_THRESHOLD = 0.27
EAR_CONSEC_FRAMES = 23

# --- Utils ---
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

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
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

phone_detector = PhoneDetector(model_path='yolov5s.pt')

COUNTER = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    is_drowsy = False
    head_direction = "Forward"
    phone_in_use = False
    ear = 0.0

    if results.multi_face_landmarks:
        mesh_points = np.array([
            [int(p.x * w), int(p.y * h)]
            for p in results.multi_face_landmarks[0].landmark
        ])

        left_eye = mesh_points[LEFT_EYE]
        right_eye = mesh_points[RIGHT_EYE]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EAR_CONSEC_FRAMES:
                is_drowsy = True
        else:
            COUNTER = 0

        # Head pose based on nose position
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
        f"EAR: {ear:.2f}" if results.multi_face_landmarks else "EAR: N/A",
        f"PERCLOS: {round(1 - ear, 2) if results.multi_face_landmarks else 'N/A'}"
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


