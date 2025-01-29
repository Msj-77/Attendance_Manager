import cv2
import os
import csv
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.model_selection import train_test_split

with open('data/names.pkl', 'rb') as file:
    labels = pickle.load(file)

with open('data/face_data.pkl', 'rb') as file:
    faces = pickle.load(file)

faces = faces.reshape(len(faces), -1)

X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

frame_w, frame_h = 640, 480
panel_w = 200

if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

frame_count = 0
interval = 1
name = "Waiting..."
last_time = "--:--:--"

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (frame_w, frame_h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ui_panel = np.ones((frame_h, panel_w, 3), dtype=np.uint8) * 50

    if frame_count % interval == 0:
        detected_faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        for (x, y, w, h) in detected_faces:
            cropped_face = frame[y:y+h, x:x+w, :]
            resized_face = cv2.resize(cropped_face, (30, 30)).flatten().reshape(1, -1)
            
            try:
                prediction = knn.predict(resized_face)
                name = str(prediction[0])
                last_time = datetime.now().strftime("%H:%M:%S")
            except:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    file_path = f"Attendance/Attendance_{datetime.now().strftime('%d-%m-%Y')}.csv"
    file_exists = os.path.isfile(file_path)

    if cv2.waitKey(10) & 0xFF == ord('o'):
        with open(file_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["NAME", "TIME"])
            writer.writerow([name, last_time])

    cv2.putText(ui_panel, "Attendance System", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(ui_panel, f"User: {name}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(ui_panel, f"Time: {last_time}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(ui_panel, "Press 'o' to Save", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(ui_panel, "Press 'q' to Quit", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    combined_display = np.hstack((frame, ui_panel))

    cv2.imshow("Attendance System", combined_display)

    if cv2.getWindowProperty("Attendance System", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Exiting...")
        break

video.release()
cv2.destroyAllWindows()
