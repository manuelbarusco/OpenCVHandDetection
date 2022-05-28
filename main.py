"""
import cv2
import numpy as np


net = cv2.dnn.readNet(r"yolov3.weights", r"yolov3.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers = net.getLayerNames()
outLayers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread("reggio.jpg")
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(outLayers)

boxes = list()
confidences = list()
class_ids = list()

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.7:
            print(confidence)
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, h, w])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# remove double recognition
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

for i in range(len(boxes)):
    if i in indexes:
        x,y,h,w = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x,y), (x+w,y+h), colors[i], 2)
        cv2.putText(img, label + "("+str(round(confidences[i],3))+")", (x,y), cv2.FONT_HERSHEY_PLAIN, 1, colors[i], 1)

cv2.imshow("photo", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
########################################################################################################################
import cv2
from YOLOv4 import YOLO

# yolo = YOLO("cross-hands-yolov4-tiny.cfg", "cross-hands-yolov4-tiny.weights", ["hand"])
yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
yolo.size = 416
yolo.confidence = 0.5

print("extracting tags for each image...")
img = cv2.imread("rgb/30.jpg")

"""
if img.endswith(".txt"):
    with open(img, "r") as myfile:
        lines = myfile.readlines()
        files = map(lambda x: os.path.join(os.path.dirname(img), x.strip()), lines)
else:
    files = sorted(glob.glob("%s/*.jpg" % img))
"""

conf_sum = 0
detection_count = 0

width, height, inference_time, results = yolo.inference(img)
output = []

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 848, 640)

for detection in results:
    id, name, confidence, x, y, w, h = detection
    cx = x + (w / 2)
    cy = y + (h / 2)
    conf_sum += confidence
    detection_count += 1

    # draw a bounding box rectangle and label on the image
    color = (255, 0, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    text = "%s (%s)" % (name, round(confidence, 2))
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    print("%s with %s confidence" % (name, round(confidence, 2)))

    # cv2.imwrite("export.jpg", mat)

    # show the output image
    cv2.imshow('image', img)
    cv2.waitKey(0)

print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
cv2.destroyAllWindows()
