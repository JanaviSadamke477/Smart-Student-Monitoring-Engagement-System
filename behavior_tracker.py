import cv2
import dlib
from scipy.spatial import distance as dist

class BehaviorTracker:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """Initializes dlib face detector and 68-point facial landmark predictor."""
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
        except Exception as e:
            print(f"❌ Error loading dlib shape predictor: {e}")
            print("Make sure 'shape_predictor_68_face_landmarks.dat' is in your project folder!")

        # Landmark indexes in dlib 68-point model
        self.RIGHT_EYE = list(range(36, 42))
        self.LEFT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))

    def eye_aspect_ratio(self, eye):
        """Calculates Eye Aspect Ratio (EAR)."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0

    def mouth_aspect_ratio(self, coords):
        """Calculates Mouth Aspect Ratio (MAR) using inner lip coordinates."""
        # Inner lip vertical points: (61, 67), (62, 66), (63, 65)
        A = dist.euclidean(coords[61], coords[67])
        B = dist.euclidean(coords[62], coords[66])
        C = dist.euclidean(coords[63], coords[65])
        # Inner lip horizontal points: (60, 64)
        D = dist.euclidean(coords[60], coords[64])
        return (A + B + C) / (2.0 * D) if D > 0 else 0

    def analyze_frame(self, frame):
        """
        Analyzes a single video frame for behaviors.
        Returns: processed_frame, ear, mar, posture, is_yawning, is_drowsy
        """
        if frame is None:
            return None, 0, 0, "No Frame", False, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if len(rects) == 0:
            return frame, 0, 0, "No Face Detected", False, False

        # Process the first detected face
        rect = rects[0]
        shape = self.predictor(gray, rect)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Extract eye landmarks
        left_eye = [coords[i] for i in self.LEFT_EYE]
        right_eye = [coords[i] for i in self.RIGHT_EYE]

        # Calculate EAR & MAR
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = self.mouth_aspect_ratio(coords)

        # --- CALIBRATED BEHAVIORAL THRESHOLDS ---
        # EAR drops below 0.16 when eyes are closed
        is_drowsy = ear < 0.16
        # MAR exceeds 0.75 during an actual yawn
        is_yawning = mar > 0.75

        # Head Orientation Detection (Nose Tip Position vs Face Bounding Box)
        nose_x = coords[30][0]
        face_left = rect.left()
        face_right = rect.right()
        face_width = face_right - face_left

        relative_nose_pos = (nose_x - face_left) / face_width if face_width > 0 else 0.5

        if relative_nose_pos < 0.35:
            posture = "Looking Right"
        elif relative_nose_pos > 0.65:
            posture = "Looking Left"
        else:
            posture = "Attentive (Forward)"

        # Draw green tracking dots on eyes and mouth
        for (x, y) in coords[36:68]:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw face bounding box
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 2)

        return frame, ear, mar, posture, is_yawning, is_drowsy