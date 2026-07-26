import cv2
import sys
from behavior_tracker import BehaviorTracker

def open_camera():
    """Tries multiple camera indices and backends to force camera opening on Windows."""
    # Attempt 1: Index 0 with DirectShow (fastest on Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    
    # Attempt 2: Index 0 standard
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap

    # Attempt 3: Index 1 DirectShow (External or secondary camera)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap

    return None

def main():
    print("⏳ Initializing Behavior Tracker Engine...")
    tracker = BehaviorTracker()

    print("🎥 Opening Webcam...")
    cap = open_camera()

    if cap is None or not cap.isOpened():
        print("❌ ERROR: Could not open webcam.")
        print("Troubleshooting steps:")
        print("1. Close Zoom, Teams, Skype, or Windows Camera app if open.")
        print("2. Check Windows Settings -> Privacy -> Camera access is ON.")
        sys.exit(1)

    print("✅ Camera opened successfully! Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Warning: Empty frame received from camera.")
            continue

        # Process frame
        frame, ear, mar, posture, is_yawning, is_drowsy = tracker.analyze_frame(frame)

        # --- ON-SCREEN DISPLAY ---
        # Header Info
        cv2.putText(frame, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"EAR (Eyes): {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR (Yawn): {mar:.2f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Behavioral Warning Banners
        if is_drowsy:
            cv2.putText(frame, "ALERT: Eyes Closed / Drowsy!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if is_yawning:
            cv2.putText(frame, "ALERT: Yawning Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show Output Window
        cv2.imshow("Smart Student Behavior Monitor", frame)

        # Press 'q' to safely exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Camera session ended cleanly.")

if __name__ == "__main__":
    main()