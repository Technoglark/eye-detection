import cv2
import numpy as np
from matplotlib import pyplot as pl

smdetect = cv2.CascadeClassifier("eye.xml")



cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 920)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smile = smdetect.detectMultiScale(gray, 3, 2)
    print(smile)
    for (x, y, w, h) in smile:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print(smile)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
