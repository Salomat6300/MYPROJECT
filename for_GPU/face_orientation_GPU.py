# deepGPUFolder/face_orientation.py
import cv2
import mediapipe as mp
import numpy as np

class FaceOrientationDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    def detect(self, frame):
        # frame expected BGR
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return 0  # No face considered invalid
        lm = results.multi_face_landmarks[0].landmark
        left_eye = lm[33]; right_eye = lm[263]; nose = lm[1]
        lx, rx = left_eye.x * w, right_eye.x * w
        ly, ry = left_eye.y * h, right_eye.y * h
        nx, ny = nose.x * w, nose.y * h
        eye_width = abs(rx - lx) + 1e-6
        eye_x_center = (lx + rx) / 2
        nose_x_offset = abs(nx - eye_x_center)
        yaw_ok = nose_x_offset < eye_width * 0.12
        eye_y_avg = (ly + ry) / 2
        vertical_dist = abs(ny - eye_y_avg)
        vertical_ratio = vertical_dist / eye_width
        pitch_ok = 0.25 < vertical_ratio < 0.7
        return 1 if (yaw_ok and pitch_ok) else 0
