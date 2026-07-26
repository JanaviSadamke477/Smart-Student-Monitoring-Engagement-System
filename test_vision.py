import cv2
from vision_tracker import VisionTracker

def main():
    tracker = VisionTracker()
    cap = cv2.VideoCapture(0)  # Open default camera
    
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("🎥 Starting standalone vision test. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Process the frame through our vision engine
        processed_frame, face_detected, count = tracker.process_frame(frame)

        # Display visual indicators on top left
        status_msg = f"Faces Tracked: {count}" if face_detected else "Status: Searching..."
        color = (0, 255, 0) if face_detected else (0, 0, 255)
        
        cv2.putText(processed_frame, status_msg, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show live feed window
        cv2.imshow("Vision Tracking Test Mode", processed_frame)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Vision test completed.")

if __name__ == "__main__":
    main()