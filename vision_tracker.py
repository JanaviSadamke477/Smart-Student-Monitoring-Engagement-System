import cv2

class VisionTracker:
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml'):
        # Load the Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def process_frame(self, frame):
        """
        Processes a single video frame.
        Returns: (frame_with_drawings, face_detected_boolean, number_of_faces)
        """
        if frame is None:
            return None, False, 0

        # Convert frame to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        face_detected = len(faces) > 0
        
        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                "Active Student", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )
            
        return frame, face_detected, len(faces)