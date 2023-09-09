import cv2
import time
import pose as pm

# cap = cv2.VideoCapture('Videos/bench.mp4')
cap = cv2.VideoCapture(0)
dim = (1080,720)
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    img = cv2.resize(img, dim)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        num = len(lmList) -1 
        # print(lmList[num])
        for i in range(num):
            cv2.circle(img, (lmList[i][1], lmList[i][2]), 2, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {str(int(fps))}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()