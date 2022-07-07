import sys
sys.path.append('..')

from detect1 import ssd_get_bounding_boxes, cascad_get_bounding_boxes, Yolo
import numpy as np
import cv2
import imutils
import uuid
import time
import random
from imutils.object_detection import non_max_suppression
import pandas as pd
from centroidtracker import CentroidTracker
from collections import OrderedDict
from trackableobject import TrackableObject
from track_object import Tracker


def writVideo(frame):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Output.avi', fourcc, 20.0, (640, 480))
    out.write(frame)


def loadVideo():
    ct = CentroidTracker(maxDisappeared=10, maxDistance=5)
    tracker = Tracker(150, 10, 15, 1)
    video_path = "video.mp4"

    track_colors = [(255, 0, 0), (10, 255, 100), (0, 200, 255), (255, 255, 255),
                    (0, 255, 255), (255, 255, 0), (0, 150, 90), (255, 150, 150),
                    (150, 150, 0), (255, 255, 100), (255, 50, 50), (100, 255, 100),
                    (90, 128, 96), (30, 130, 0), (10, 120, 30), (25, 50, 15), (13, 80, 40), (59, 232, 12)]


    print("[INFO] loading Video File...!")
    cap = cv2.VideoCapture(video_path)
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # creates a pandas data frame with the number of rows the same length as frame count
    df = pd.DataFrame(index=range(int(frames_count)))
    df.index.name = 'Frames'
    writer = None
    skip_frame_count = 0
    H = None
    W = None
    totalDown = 0
    totalUp = 0
    count = 0
    frameN = 0
    totalpeople = 0
    centers1 = []

    trackableObjects = {}
    while True:
        count += 1
        ret, frame = cap.read()
        if not cap.isOpened():
            sys.exit('Error capturing video..')
        frame = imutils.resize(frame, width=500)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        if ret == False:
            break
       # bound, confidenfont = cv2.FONT_HERSHEY_SIMPLEXce, claase_idx = ssd_get_bounding_boxes(frame)
        # bounds, _, _ = cascad_get_bounding_boxes(frame)
        status = "Waiting"
        boxes, confidences, classe_ids, centers, status = Yolo(frame, status)

        for i in range(len(boxes)):
            # if i in NMS_indexes:
            print("i", i)
            x, y, w, h = boxes[i]
            label = classe_ids[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 200, 0), 2)
            cv2.putText(frame, "Person", (x, y + 10), cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 0), 1)

        for i, center in enumerate(centers):
            cx = center[0]
            cy = center[1]
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

        cv2.putText(frame, "The Number of People: {}".format(len(boxes)), (10, 20), 0, 0.5, (0, 0, 255), 1)
        cv2.line(frame, (30, (H // 2)), (W - 30, (H // 2)), (0, 255, 255), 2)
        if len(centers) > 0:

            # Track object using Kalman Filter
            tracker.update(centers)

            for i in range(len(tracker.tracks)):
                status = "Tracking"
                if (len(tracker.tracks[i].trace) > 1):

                    for j in range(len(tracker.tracks[i].trace)):
                        x1 = int(tracker.tracks[i].trace[-1][0][0])
                        y1 = int(tracker.tracks[i].trace[-1][1][0])
                        top_left = (x1 - 30, y1 - 50)
                        right_bottom = (x1 + 30, y1 + 50)
                        center = np.array([[x1], [y1]])
                        centers1.append(center)

                        colors = tracker.tracks[i].track_id % 18
                        cv2.rectangle(frame, top_left, right_bottom, track_colors[colors], 1)
                        cv2.circle(frame, (x1, y1), 2, (0, 255, 150), -1)
                        cv2.putText(frame, str(tracker.tracks[i].track_id), (x1 - 20, y1 - 30), 0, 0.4, track_colors[colors], 2)

                    for k in range(len(tracker.tracks[i].trace)):
                        x = int(tracker.tracks[i].trace[k][0][0])
                        y = int(tracker.tracks[i].trace[k][1][0])
                        cv2.circle(frame, (x, y), 2, track_colors[colors], -1)

        objects = ct.update(centers)

        for (objectId, centroid) in objects.items():

            to = trackableObjects.get(objectId, None)
            # print("to", to)
            if to is None:
                to = TrackableObject(objectId, centroid)

            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                # print("direction", direction)
                to.centroids.append(centroid)

                if not to.Counted:
                    if direction < 0 and centroid[1] < (H // 2):
                        totalUp += 1
                        to.Counted = True
                       
                    elif direction > 0 and centroid[1] > (H // 2):
                        totalDown += 1
                        to.Counted = True
                        

            trackableObjects[objectId] = to
            text = "ID {}".format(objectId)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        info = [("Up", totalUp),
                ("Down", totalDown),
                ("status", status), ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 150), 2)
        frameN += 1
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


loadVideo()

