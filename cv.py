
import cv2 as c
import numpy as np
from keras.models import load_model
from cvzone.ClassificationModule import Classifier
import math
import time
from cvzone.HandTrackingModule import HandDetector
model = load_model('64x3-CNN2.model')
labels = ["1","2","3","4"]
# folder = '/home/subhash/Documents/project/newfolder3'
offset = 20
imgsize = 50
counter = 0
video = c.VideoCapture(0)
detector = HandDetector(maxHands=1)
while True:
    success,img = video.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)#,draw=False
    #img = c.cvtColor(img,c.COLOR_BGR2GRAY)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255
        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgcropshape = imgcrop.shape

        aspectratio = h/w
        if aspectratio>1:
            k = imgsize/h
            wCal = math.ceil(k*w)
            imgResize = c.resize(imgcrop,(wCal,imgsize))
            imgResizeshape = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)
            #imgwhite[0:imgResizeshape[0], 0:imgResizeshape[1]] = imgResize
            imgwhite[:,wGap:wCal+wGap] = imgResize
            imgwhite = c.cvtColor(imgwhite,c.COLOR_BGR2GRAY)
            imgwhite = np.array(imgwhite).reshape(-1, 50, 50, 1)
            prediction = model.predict(imgwhite) # draw = False
            index = int(prediction)
            print(labels[index])
        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = c.resize(imgcrop, (hCal, imgsize))
            imgResizeshape = imgResize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            # imgwhite[0:imgResizeshape[0], 0:imgResizeshape[1]] = imgResize

            imgwhite[:,hGap: hCal + hGap] = imgResize
            imgwhite = c.cvtColor(imgwhite, c.COLOR_BGR2GRAY)
            imgwhite = np.array(imgwhite).reshape(-1, 50, 50, 1)
            prediction = model.predict(imgwhite)  # draw = False
            index = int(prediction)
            print(labels[index])
        c.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),c.FILLED)
        c.putText(imgOutput,labels[index],(x,y-20),c.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        c.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        #c.imshow("ImageWhite", imgwhite)

    c.imshow("Image",img)
    if c.waitKey(1) == ord('q'):
        break
    #      counter += 1
    #      c.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
    #      print(counter)


video.release()
c.destroyAllWindows()