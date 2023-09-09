from __future__ import print_function # Library for printing the result of recording FPS
from imutils.video import WebcamVideoStream # Library for Webcam Classes
from imutils.video import FPS #Library for Recording FPS
import imutils #Library Hack to boost FPS by using extra CPU for Image Processing

import cv2
import time
import pose as pm

# cap = cv2.VideoCapture('Videos/bench.mp4')
#camera
# cap = cv2.VideoCapture(0)
cap = WebcamVideoStream(src=0).start()
dim = (1080,720)

fps = FPS().start()


pTime = 0
detector = pm.poseDetector()
while True and fps._numFrames != -1:
    # success, img = cap.read()
    img = cap.read()
    img = detector.findPose(img)
    img = cv2.resize(img, dim)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        num = len(lmList) -1 
        # print(lmList[num])
        for i in range(num):
            cv2.circle(img, (lmList[i][1], lmList[i][2]), 2, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    current_fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f"FPS: {str(int(current_fps))}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    fps.update()
    if key == ord("q"):
        break
    
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()