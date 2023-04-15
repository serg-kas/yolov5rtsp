import torch
import torch.nn as nn
import cv2 as cv
import threading
import os
from time import sleep

#
RTSP_URL = 'rtsp://admin:daH_2019@192.168.5.44:554/cam/realmonitor?channel=13&subtype=1'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;udp"

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 'Upsample' object has no attribute 'recompute_scale_factor'
for m in model.modules():
    if isinstance(m, nn.Upsample):
        m.recompute_scale_factor = None

#
THICKNESS = 2
COLOR = (255, 0, 0)

#
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

names_to_detect = names
# names_to_detect = ['person']
classes_to_detect = []
for idx, name in enumerate(names):
    if name in names_to_detect:
        classes_to_detect.append(idx)
print("Распознаем классы (id) {}:".format(classes_to_detect))


class MyThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.frame = None
        self.stop = False

    def run(self):
        while not self.stop:
            # wait if no frame
            if self.frame is None:
                sleep(0.01)
                continue

            #
            results = model([self.frame]).xyxy[0]

            #
            result_list = []
            for row in results:
                cl = int(row[-1])
                conf = float(row[-2])
                coords = tuple(row.int().numpy()[:-2])
                #
                if cl in classes_to_detect:
                    result_list.append([coords, conf, cl])
                #
                self.result = result_list
                self.frame = None


#
cap = cv.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise IOError("Cannot open cam")

#
myThread = MyThread()
myThread.start()

#
while True:
    # получаем новый фрейм
    ret, frame = cap.read()

    # засылаем новый фрейм на предикт
    if myThread.frame is None:
        myThread.frame = frame

    # TODO: рисовать bb разными цветами
    if myThread.result is not None:
        # print(len(myThread.result))
        for res in myThread.result:
            (X1, Y1, X2, Y2), _, cl = res
            # frame = cv.rectangle(frame, myThread.result[:2], myThread.result[2:], COLOR, THICKNESS)
            frame = cv.rectangle(frame, (X1, Y1), (X2, Y2), COLOR, THICKNESS)
            frame = cv.putText(frame, names[cl], (X1, Y1 + 10), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)

    #
    cv.imshow(RTSP_URL, frame)

    #
    c = cv.waitKey(1)
    if c == 27:
        myThread.stop = True
        break

#
cap.release()
cv.destroyAllWindows()


