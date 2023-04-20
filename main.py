import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import threading
import os
from time import sleep
#
import utils_small as u
#
# colors = [u.blue, u.green, u.red, u.white, u.yellow, u.purple, u.turquoise, u.black]
colors = [tuple(np.random.choice(range(256), size=3)) for x in range(80)]
colors = [(int(colors[x][0]), int(colors[x][1]), int(colors[x][2])) for x in range(80)]
#
RTSP_URL = 'rtsp://admin:daH_2019@192.168.5.44:554/cam/realmonitor?channel=13&subtype=1'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;udp"

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# HOOK: prevent error 'Upsample' object has no attribute 'recompute_scale_factor'
for m in model.modules():
    if isinstance(m, nn.Upsample):
        m.recompute_scale_factor = None

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

classes_list = []
for idx, name in enumerate(names):
    if name in names_to_detect:
        classes_list.append(idx)
print("Распознаем классов: {}".format(len(classes_list)))


class MyThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.frame = None
        self.stop = False
        #
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # # HOOK: prevent error 'Upsample' object has no attribute 'recompute_scale_factor'
        # for m in self.model.modules():
        #     if isinstance(m, nn.Upsample):
        #         m.recompute_scale_factor = None

    def run(self):
        while not self.stop:
            # wait if no frame
            if self.frame is None:
                print("Нет фрейма для предикта")
                # sleep(0.01)
                sleep(0.05)
                continue
            #
            # results = self.model([self.frame]).xyxy[0]
            results = model([self.frame]).xyxy[0]
            #
            result_list = []
            for row in results:
                coords = tuple(row.int().numpy()[:-2])
                conf = float(row[-2])
                curr_class = int(row[-1])
                #
                if curr_class in classes_list:
                    result_list.append([coords, conf, curr_class])
            #
            if len(result_list) > 0:
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
    if ret:
        print("Получен новый фрейм")
        # sent_frame_for_pred = False
        # засылаем новый фрейм на предикт
        if myThread.frame is None:
            myThread.frame = frame

            print("Новый фрейм на предикт")
            # sent_frame_for_pred = True
        else:
            print("Нейронка фрейм не берет")
    else:
        print("Нет нового фрейма")

    #
    if myThread.result is not None:
        print("Получен предикт, классов:", len(myThread.result))
        for res in myThread.result:
            # print("res", res)
            (X1, Y1, X2, Y2), _, class_id = res
            COLOR = colors[class_id]
            frame = cv.rectangle(frame, (X1, Y1), (X2, Y2), COLOR, thickness=2)
            frame = cv.putText(frame, names[class_id], (X1, Y1 + 10), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=COLOR, thickness=1)
    else:
        print("Предикт не готов")

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


