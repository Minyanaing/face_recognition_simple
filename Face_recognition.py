import cv2
import face_recognition

import numpy as np
import os
from datetime import datetime

imgPath = "Face_imgs"
images = []
names = []
myList = os.listdir(imgPath)
for i in myList:
    img = cv2.imread(f"{imgPath}/{i}")
    images.append(img)
    names.append(os.path.splitext(i)[0])
print(names)

# encoding for already know faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# adding name and time to CSV
def addCSV(name):
    if os.path.exists("./NameList.csv"):
        f = open("./NameList.csv", "r+")
        myData = f.readlines()
        nameList = []
        # find all the name in  CSV
        for line in myData:
            entry = line.split(',')
            nameList.append(entry[0])
        # if found name is not in CSV, add name
        if name not in nameList:
            currentTime = datetime.now()
            dtString = currentTime.strftime('%H:%M:%S')
            f.writelines(f'{name},{dtString}\n')
            f.close()
    else:
        f = open("./NameList.csv", "w")
        f.write("Name,Time\n")
        f.close()

encodedKnownFaces = findEncodings(images)
print("Encoding finish")

# webCam open
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # scaling 1/4 of size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # finding face location and encoding face-measurements
    faceCam = face_recognition.face_locations(imgS) # return y1,x2,y2,x1 for square box
    encodeCam = face_recognition.face_encodings(imgS, faceCam)

    for encodeFace, face in zip(encodeCam, faceCam):
        # use Linear SVM at the backend to find matching
        matches = face_recognition.compare_faces(encodedKnownFaces, encodeFace)

        # return distance with each face (lowest will be best match)
        faceDistance = face_recognition.face_distance(encodedKnownFaces, encodeFace)
        matchIndex = np.argmin(faceDistance) # return index value of min: value

        if matches[matchIndex]: # mathces return FALSE if not found mathced face.
            name = names[matchIndex]
            addCSV(name) # add name and time to CSV

            y1,x2,y2,x1 = face
            # re-size back to original, since imshow() on original image
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1,y2-25), (x2,y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.imshow("WebCam", img)
    cv2.waitKey(1)