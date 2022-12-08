import cv2
import numpy as np
import pickle

#cascade classifier
face_cascade = cv2.CascadeClassifier('cascade\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
   og_labels = pickle.load(f) 
   labels = {v:k for k,v in og_labels.items()}

print(og_labels)
print (labels)

#Open webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    #detecting faces
    for (x,y,w,h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # recognize use deep learn models
        #print(recognizer.predict(roi_gray))
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            names = labels[id_]
            color = (255,255,255)
            stroke = 4
            cv2.putText(frame, names, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        

        color = (255,0,0) #BGR
        stroke = 4
        endcord_x = x + w
        endcord_y = y + h
        cv2.rectangle(frame, (x,y),(endcord_x,endcord_y),color,stroke)

    #Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
