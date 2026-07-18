import cv2
import urllib.request
import os

# 1. XML file ko direct tumhare folder me download karne ka intazam
xml_filename = "haarcascade_frontalface_default.xml"

if not os.path.exists(xml_filename):
    print("Face detection file download ho rahi hai... Kripya ek second rukein.")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_filename)
    print("Download complete!")

# 2. Ab local downloaded file se Face Detector load karo
face_cascade = cv2.CascadeClassifier(xml_filename)

# 3. Start Laptop Webcam
cap = cv2.VideoCapture(0)

print("Webcam chalu ho raha hai... Band karne ke liye keyboard par 'q' dabayein.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Webcam se feed nahi aa rahi hai.")
        break

    # Gray image convert karo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detect karo
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Agar face detect hota hai toh box banao
    if len(faces) > 0:
        print(f"Face Detected! Total faces: {len(faces)}")
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Screen par live video window show hogi
    cv2.imshow('Smart Student Monitor - Local Camera Test', frame)
    
    # 'q' key dabane par close hoga
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam successfully closed.")