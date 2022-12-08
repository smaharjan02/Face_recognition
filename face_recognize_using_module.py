import cv2
import face_recognition as fr
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Faces")
video_capture = cv2.VideoCapture(0)


current_id = 0
labels = []
known_face_encodings = []
known_face_names = []


# def unknown_name(name,frame)
def get_images():
#getting face encodings and face names from faces directory
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root,file)
                label = os.path.basename(os.path.dirname(path)).replace (" ","-").lower()
            
                known_face_names.append(label)

                images = fr.load_image_file(path)
            
                face_encoding =fr.face_encodings(images)[0]

                known_face_encodings.append(face_encoding)


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

get_images()

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25,fy=0.25) 
    rgb_small_frame = small_frame[:,:,::-1]
    if process_this_frame:
        #finding all the faces in the frame
        face_locations = fr.face_locations(rgb_small_frame)
        #finding all face encodings for the face in frame
        face_encodings = fr.face_encodings(rgb_small_frame,face_locations)

        face_names = []
        
        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings,face_encoding, tolerance=0.6)
            name = "unknown"
            face_distances = fr.face_distance(known_face_encodings,face_encoding)
            
            best_match_index = np.argmin(face_distances)
            #print("Best_Match_Index: ",best_match_index)
            print(f"Distance between frame and each known image: ",{best_match_index})
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            

            

    process_this_frame = not process_this_frame
    print("Face detected -- {}".format(face_names))
    for (t,r,b,l), name in zip (face_locations, face_names):
        t *=4
        r *=4
        b *=4
        l *=4

        cv2.rectangle(frame, (l,t), (r,b), (0,255,0),2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name, (l+6,b-6), font, 0.6, (255,255,255),1)
    
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



