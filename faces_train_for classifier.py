import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascade\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer.create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace (" ","-").lower()
            # print(label,path)
            # y_labels.append(label) # some label
            # x_train.append(path) # verify this image and turn into a numpy array, GRAY
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1   
            id_ = label_ids[label]
            #print(label_ids)
            #print(path)
            
            pil_image = Image.open(path).convert("L") #grayscale
            #print(pil_image)
            size = (550,550)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            #print("Final Image",pil_image.show())
            image_array = np.array(final_image,"uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor = 1.1, minNeighbors= 5)
            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                #print(roi)
                x_train.append(roi)
                y_labels.append(id_)

                # color = (255,0,0) #BGR
                # stroke = 4
                # endcord_x = x + w
                # endcord_y = y + h
                # cv2.rectangle(roi, (x,y),(endcord_x,endcord_y),color,stroke)
        #print(len(faces))

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids,f)     

#print(x_train)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")