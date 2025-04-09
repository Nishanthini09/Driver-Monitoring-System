import cv2
import numpy as np
import mediapipe as mp

class DrowsinessDetector:
    def __init__(self, ear_thresh=0.25, consec_frames=20):
        self.EAR_THRESHOLD = ear_thresh
        self.EAR_CONSEC_FRAMES = consec_frames
        self.counter = 0
        self.is_drowsy = False

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [263, 387, 385, 362, 380, 373]

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ear = None
        mesh_points = None

        if results.multi_face_landmarks:
            mesh_points = np.array([
                [int(p.x * w), int(p.y * h)]
                for p in results.multi_face_landmarks[0].landmark
            ])

            left_eye = mesh_points[self.LEFT_EYE]
            right_eye = mesh_points[self.RIGHT_EYE]

            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Draw eyes
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

            if ear < self.EAR_THRESHOLD:
                self.counter += 1
                if self.counter >= self.EAR_CONSEC_FRAMES:
                    self.is_drowsy = True
            else:
                self.counter = 0
                self.is_drowsy = False

        return frame, ear, self.is_drowsy, mesh_points
