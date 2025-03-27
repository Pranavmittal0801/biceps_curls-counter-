import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("C:\\Users\\HP\\OneDrive\\Desktop\\pose\\curls.mp4")
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

detector = pm.poseDetector()
count = 0
dir = 0  # 0 = Going Down, 1 = Going Up
pTime = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Video has ended or frame capture failed")
        break

    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Right Arm (Shoulder - Elbow - Wrist)
        angle = detector.findAngle(img, 12, 14, 16)

        # **✅ Reverse Mapping so 100% is at the top**
        per = np.interp(angle, (50, 160), (100, 0))  # Now 50° = 100% (Top), 160° = 0% (Bottom)
        bar = np.interp(angle, (50, 160), (100, 650))  

        # **✅ Fix: Reverse Color Logic**
        if per >= 98:  # Fully curled (100%)
            color = (0, 255, 0)  # Green ✅ (Correct)
            if dir == 0:  # Moving down -> Now fully curled
                count += 1
                dir = 1
                print(f"✅ Rep Counted: {count}")

        elif per <= 5:  # Fully extended (0%)
            color = (255, 0, 255)  # Pink 🟣 (Correct)
            if dir == 1:  # Moving up -> Now fully extended
                dir = 0

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), (255, 0, 255), 3)  # Border always Pink
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)  # Filled bar changes color
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
