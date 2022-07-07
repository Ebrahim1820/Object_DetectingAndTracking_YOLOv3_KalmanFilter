
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def cascad_get_bounding_boxes(frame):
    frame = frame
    print("[INFO] loading Cascade_Model detection...!")
    fullbody_cascade = cv2.CascadeClassifier("./case.xml")
    # print(fullbody_cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bounding_boxes = fullbody_cascade.detectMultiScale(gray, 1.3, 5)
    # print(bounding_boxes)

    for x, y, w, h in bounding_boxes:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # cascad
    #cv2.imshow("frameCas", frame)
    return bounding_boxes, None, None


def ssd_get_bounding_boxes(frame):
    frame = frame
    # global box
    bounding_boxes = []
    prototxt = os.path.join(__location__, 'MobileNetSSD_deploy.prototxt')
    model = os.path.join(__location__, 'MobileNetSSD_deploy.caffemodel')
    #print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    IGNORE = set(["background", "aeroplane", "bird", "boat",
                  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "pottedplant", "sheep",
                  "sofa", "train", "tvmonitor"])
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    box = []
    cls_Id = None
    # The output detections is a 4-D matrix,
    # The 3rd dimension iterates over the detected objects.
    #(i is the iterator over the number of objects)
    # The fourth dimension contains information about the bounding box and
    # score for each object. For example,detections[0,0,0,2] gives
    # the confidence score for the first object, and detections[0,0,0,3:6] give
    # the bounding box
    #[0., 15., 0.43252543,  0.6875222, 0.29533893, 0.77618074,  0.43401772]
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        # if confidence > args["confidence"]:
        if confidence > 0.5:
                # extract the index of the class label from the
                # `detections`
            idx = int(detections[0, 0, i, 1])

            # if the predicted class label is in the set of classes
            # we want to ignore then skip the detection
            if CLASSES[idx] in IGNORE:
                continue
                # bounding box are normalized between [0,1].So the coordinates
                # should be multiplied by the height and width of the original image
                # to get correct bounding box or
                # compute the (x, y)-coordinates of the bounding box for
                # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")
            bounding_boxes.append(box.astype("int"))
            cls_Id = CLASSES[idx]

    return bounding_boxes, confidence, cls_Id


def Yolo(frame):

    #yolo_Weights = os.path.join(__location__, './yolov3-tiny.weights')
    yolo_Weights = os.path.join(__location__, './yolov3.weights')
    yolo_cfg = os.path.join(__location__, './yolov3.cfg')
    coco_data = os.path.join(__location__, './coco.names')

    net = cv2.dnn.readNet(yolo_Weights, yolo_cfg)
    h, w = frame.shape[:2]
    classes = []

    with open(coco_data, "r") as f:
        classes = [line.strip() for line in f.readlines() if line == 'person']

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    Colors = np.random.uniform(0, 255, size=(len(classes), 3))
    count = 0
    boxes = []
    centers = []
    confidences = []
    class_ids = []
    for detections in outs:
        for detect in detections:

            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:

                if class_id != 0:
                    continue
                   # Object has been detected
                center_x = int(detect[0] * w)
                center_y = int(detect[1] * h)

                width = int(detect[2] * w)
                height = int(detect[3] * h)

                # create rectange and center point obejct
                # formula for extract top left and bottom right for create rectangel
                # is x = ( center_x - width / 2) and y = (center_y - height / 2)
                left_x = int(center_x - width / 2)  # x and y top left
                top_y = int(center_y - height / 2)

                x1 = int(left_x + width)  # x and y right down
                y1 = int(top_y + height)

                center = np.array([[center_x], [center_y]])
                centers.append(np.round(center))

                boxes.append([left_x, top_y, width, height])  # save rectangels of each object in the lsit
                confidences.append(confidences)  # save the confidence of each object in list
                class_ids.append(class_id)  # save the name of each object here is person


    return boxes, confidences, class_ids, centers


video_path = "videoplayback1.mp4"
cap = cv2.VideoCapture(video_path)
H = None
W = None
savePath = '../'
count, frame_cnt = -1, 0
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

    if count >= 200 and count % 15 == 0:

        boxes, _, _, centers = Yolo(frame)
       #boxes, _, _ = cascad_get_bounding_boxes(frame)
        #boxes, _, _ = ssd_get_bounding_boxes(frame)
        # print(len(centers))
        for i in range(len(boxes)):
            [x, y, w, h] = boxes[i]
            print("box", [x, y, w, h])
            res = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

            #cv2.imshow('frame', frame)

        cv2.imwrite(savePath+"Frame%d.jpg" % count, frame)
        frame_cnt += 1
        print(count)
    #cv2.imshow('frame', frame)
    #print("Number of people ", len(boxes))
    #print("object ", class_ids)
    key = cv2.waitKey(25) & 0xFF
    if frame_cnt >= 30:
        break
    if key == ord('q'):
        break
cap.release()
# writer.release()
cv2.destroyAllWindows()

#NMS_indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0, 3)
# print("NMS_indexes", NMS_indexes)

#num_object_detected = len(boxes)
#font = cv2.FONT_HERSHEY_SIMPLEX
# for i in range(len(boxes)):

#x, y, w, h = boxes[i]
#label = classes[class_ids[i]]
#color = Colors[i]
#print("label", label)
#cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

#cv2.putText(frame, label, (x - 10, y - 10), font, .5, color, 2)
#(x1 - 20, y1 - 30), 0, 0.4
