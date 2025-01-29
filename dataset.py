import cv2
import pickle
import os
import numpy as np

vid = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_data = []
name = input("Enter User's name: ")

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized = cv2.resize(crop_img, (30, 30)) 
        
        if len(face_data) < 15: 
            face_data.append(resized)
            cv2.putText(frame, str(len(face_data)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (69, 69, 69), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (28, 32, 109), 1)
    
    cv2.imshow("Camera", frame)

    if len(face_data) == 15: 
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


face_data = np.array(face_data)
face_data = face_data.reshape((15, -1)) 


if not os.path.exists('data/'):
    os.makedirs('data/')


if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as face:
        pickle.dump(face_data, face)
else:
    with open('data/face_data.pkl', 'rb') as face:
        faces = pickle.load(face)
    faces = np.append(faces, face_data, axis=0)
    with open('data/face_data.pkl', 'wb') as face:
        pickle.dump(faces, face)


if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 15  # 
    with open('data/names.pkl', 'wb') as face:
        pickle.dump(names, face)
else:
    with open('data/names.pkl', 'rb') as face:
        names = pickle.load(face)
    names = names + [name] * 15
    with open('data/names.pkl', 'wb') as face:
        pickle.dump(names, face)

print(f"Data collection complete for {name} with 15 images!")
