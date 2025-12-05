import cv2
import numpy as np
import math
from IPython.display import display, clear_output, Image as IPythonImage
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

MIN_HAND_AREA = 2000
SAFE_DIST = 150
WARNING_DIST = 60

def point_to_rect_distance(px, py, rect):
    x1, y1, x2, y2 = rect
    if px < x1: dx = x1 - px
    elif px > x2: dx = px - x2
    else: dx = 0

    if py < y1: dy = y1 - py
    elif py > y2: dy = py - y2
    else: dy = 0

    return math.sqrt(dx*dx + dy*dy)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
else:
    print(" Camera running... Move your hand near box.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        
        rw, rh = 120, 120
        rx1, ry1 = w//2 - rw//2, h//2 - rh//2
        rx2, ry2 = rx1 + rw, ry1 + rh
        rect = (rx1, ry1, rx2, ry2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 40, 60])
        upper = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            mask[fy:fy+fh, fx:fx+fw] = 0  # Remove face from candidate hand region

        
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_center = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > MIN_HAND_AREA:

                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                    hand_center = (cx, cy)
                    cv2.circle(frame, hand_center, 8, (0,0,255), -1)

        if hand_center:
            dist = point_to_rect_distance(hand_center[0], hand_center[1], rect)
            if dist > SAFE_DIST:
                state, col = "SAFE", (0,255,0)
            elif dist > WARNING_DIST:
                state, col = "WARNING", (0,255,255)
            else:
                state, col = "DANGER", (0,0,255)
        else:
            state, col = "SAFE", (0,255,0)

        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col, 3)
        cv2.putText(frame, state, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

        if state == "DANGER":
            cv2.putText(frame, " DANGER DANGER ", (60, h-30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3)

        _, buffer = cv2.imencode(".jpg", frame)
        clear_output(wait=True)
        display(IPythonImage(data=buffer.tobytes()))

        time.sleep(0.05)

except KeyboardInterrupt:
    print(" Stopped.")

cap.release()
