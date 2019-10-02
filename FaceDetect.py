# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (224, 224)

face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture('video.mp4')
img = cv2.imread('imagem/serie.jpg')

i = 1
j = 1
while(cap.isOpened()):
    ret, frame = cap.read()
  
    if ret == True:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
        )

        crop_img = 0
        
        plt.hist(frame.ravel(),256,[0,256])
        histg = cv2.calcHist([frame],[0],None,[256],[0,256])
        plt.savefig("Hist/hist_"+str(j)+".png", dpi=200)
        j += 1

        for (x,y,w,h) in faces:
            frame_c = cv2.rectangle(frame,(x,y),(x+w,y+h),2)
            crop_img = frame_c[y:y+h, x:x+w]
            cv2.imwrite("Pessoas/face_"+str(i)+".jpg", crop_img)

            plt.hist(crop_img.ravel(),256,[0,256])
            histg = cv2.calcHist([crop_img],[0],None,[256],[0,256])
            plt.savefig("HistFace/hist_"+str(i)+".jpg")

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            i +=1
            

    else: 
        break

    #cv2.imwrite("imagem/serie_face.jpg", crop_img)

#cv2.imshow('img',img)
cap.release()
cv2.destroyAllWindows()








