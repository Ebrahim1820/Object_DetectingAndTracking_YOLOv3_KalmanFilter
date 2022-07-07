import cv2
import numpy as np
import imutils
from imutils import contours
import time
import copy
from track_object import Tracker
import random
from random import randint
from detect1 import ssd_get_bounding_boxes, cascad_get_bounding_boxes, Yolo

count = 0

bgs = cv2.bgsegm.createBackgroundSubtractorMOG()
#bgs = cv2.createBackgroundSubtractorMOG2()
#path = "D:/Code/video/Crowded.mp4"
path = "D:/Code/video/videoplayback1.mp4"
cap = cv2.VideoCapture(path)

#tracker = Tracker(160, 30, 10, 100)
tracker = Tracker(160, 40, 20, 100)
#tracker = Tracker(150, 30, 5)

skip_frame_count = 0
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255),
                (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 150, 150),
                (150, 150, 0), (100, 0, 100), (255, 50, 50), (100, 255, 100),
                (50, 50, 50), (100, 100, 100), (0, 120, 200)]


while True:

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    time.sleep(.100)
    if ret == False:
        break
    orig_frame = copy.copy(frame)

    bound, confidence, claase_idx, cnts = Yolo(frame)

    for (x, y, w, h) in bound:
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)

        cx = int(x + w / 2)
        cy = int(y + h / 2)
        print("cx", cx)
        print("cy", cy)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
    centers = []
    # # print(cnts)
    for (i, c) in enumerate(cnts):

        #     print("c", c)
        if cv2.contourArea(c) < 100:
            continue
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centeroid = (cx, cy)
        else:
            cx, cy = 0, 0
    #     #cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

        b = np.array([[cx], [cy]])
        centers.append(np.round(b))

    #  # If centroids are detected then track them
    if (len(centers) > 0):

        # Track object using Kalman Filter
        tracker.update(centers)

        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            if (len(tracker.tracks[i].trace) > 1):
                for j in range(len(tracker.tracks[i].trace)):
                    x1 = int(tracker.tracks[i].trace[-1][0][0])
                    y1 = int(tracker.tracks[i].trace[-1][1][0])
                #x2 = tracker.tracks[i].trace[j + 1][0][0]
                #y2 = tracker.tracks[i].trace[j + 1][1][0]
                    tl = (x1 - 12, y1 - 30)
                    br = (x1 + 12, y1 + 30)
                    cv2.rectangle(frame, tl, br, track_colors[i], 1)
                    cv2.putText(frame, str(tracker.tracks[i].track_id), (x1 - 20, y1 - 30), 0, 0.4, track_colors[i], 2)
                for j in range(len(tracker.tracks[i].trace)):
                    x = int(tracker.tracks[i].trace[j][0][0])
                    y = int(tracker.tracks[i].trace[j][1][0])
                    cv2.circle(frame, (x, y), 2, track_colors[i], -1)
                cv2.circle(frame, (x, y), 3, track_colors[i], -1)
           # cv2.circle(frame,())
                #colors = tracker.tracks[i].track_id % 15
                #cv2.line(frame, (int(x1), int(y1)), (int(x), int(y)), track_colors[i], 3)

    # cv2.imshow("thresh2", fgdilated)
    cv2.imshow("Frame", frame)
    #print("\n Centers: ", centers, "\n")
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
