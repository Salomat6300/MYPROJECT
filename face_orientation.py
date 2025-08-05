import cv2
import mediapipe as mp

class FaceOrientationDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    def detect(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return "No face", None

        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]

        left_eye_x = left_eye.x * w
        right_eye_x = right_eye.x * w
        left_eye_y = left_eye.y * h
        right_eye_y = right_eye.y * h
        nose_x = nose.x * w
        nose_y = nose.y * h

        eye_width = abs(right_eye_x - left_eye_x)
        eye_x_center = (left_eye_x + right_eye_x) / 2
        nose_x_offset = abs(nose_x - eye_x_center)
        yaw_ok = nose_x_offset < eye_width * 0.08

        eye_y_avg = (left_eye_y + right_eye_y) / 2
        vertical_dist = abs(nose_y - eye_y_avg)
        vertical_ratio = vertical_dist / eye_width
        pitch_ok = 0.3 < vertical_ratio < 0.6

        orientation = 1 if yaw_ok and pitch_ok else 0

        return orientation
