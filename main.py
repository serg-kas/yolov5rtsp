import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import threading
import os
from random import randint
from time import sleep
#
import utils_small as u

#
DEBUG = False
#
colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for x in range(80)]
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
#
names_to_detect = names
# names_to_detect = ['person']
classes_list = []
for idx, name in enumerate(names):
    if name in names_to_detect:
        classes_list.append(idx)
if DEBUG:
    print("Распознаем классов: {}".format(len(classes_list)))

#
RTSP_URL = 'rtsp://admin:daH_2019@192.168.5.44:554/cam/realmonitor?channel=7&subtype=1'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;udp"
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# # HOOK: prevent error 'Upsample' object has no attribute 'recompute_scale_factor'
# for m in model.modules():
#     if isinstance(m, nn.Upsample):
#         m.recompute_scale_factor = None


class MyThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.image = None
        self.stop = False
        #
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # self.model.conf = 0.25  # confidence threshold (0-1)
        # self.model.iou = 0.45  # NMS IoU threshold (0-1)

        # HOOK: prevent error 'Upsample' object has no attribute 'recompute_scale_factor'
        for m in self.model.modules():
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None

    def run(self):
        while not self.stop:
            # wait if no image
            if self.image is None:
                if DEBUG:
                    print("Нейронка свободна, но фрейма для предикта нет")
                sleep(0.01)
                # sleep(1)
                # sleep(0.5)
                continue
            #
            results = self.model([self.image]).xyxy[0]
            # results = model([self.image]).xyxy[0]

            result_list = []
            for row in results:
                coords = tuple(row.int().numpy()[:-2])
                conf = float(row[-2])
                curr_class = int(row[-1])
                if curr_class in classes_list:
                    result_list.append([coords, conf, curr_class, -1])  # -1 - номер трека по умолчанию

            self.result = result_list
            self.image = None


#
cap = cv.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise IOError("Cannot open cam")
else:
    if DEBUG:
        print("Подключили capture: {}".format(RTSP_URL))
#
myThread = MyThread()
myThread.start()

#
N = 15  # размер аккумулятора, предиктов
n = 3  # сколько детекций объекта в аккумуляторе считаем достоверной детекцией
assert n <= N

#
track = 0
#
acc_result = []   # аккумулируемый результат
while True:
    # получаем новый фрейм
    ret, frame = cap.read()
    if ret:
        if DEBUG:
            print("Получен новый фрейм")

        # засылаем новый фрейм на предикт
        if myThread.image is None:
            myThread.image = frame.copy()
            if DEBUG:
                print("Новый фрейм подан на предикт")
        else:
            if DEBUG:
                print("Нейронка занята, фрейм не берет")
    else:
        if DEBUG:
            print("Нет нового фрейма")

    #
    if myThread.result is not None:
        if DEBUG:
            print("Получен предикт, классов:", len(myThread.result))
        #
        result = []
        for new_obj in myThread.result:
            #
            obj_recognized = False
            for pred in acc_result:
                for obj in pred:
                    #
                    if new_obj[2] == obj[2] and u.get_iou(new_obj[0], obj[0]) > 0.80:
                        obj_recognized = True
                        if obj[3] == -1:
                            print("Двум идентичным объектам {} и {} присвоен новый трек {}".format(names[obj[2]],
                                                                                                   names[new_obj[2]],
                                                                                                   track))
                            obj[3] = track
                            new_obj[3] = track
                            track += 1 if track < 1000 else 0
                        else:
                            print("Новому объекту {} присвоен существующий трек".format(names[obj[2]], obj[3]))
                            new_obj[3] = obj[3]
                        break
                if obj_recognized:
                    break

            if not obj_recognized:
                print("Новому объекту {} присвоен новый трек".format(names[new_obj[2]], track))
                new_obj[3] = track
                track += 1 if track < 1000 else 0
            #
            result.append(new_obj)
        #
        acc_result.append(result)
        if len(acc_result) > N:
            acc_result.pop(0)

        #
        result_np = []
        for pred in acc_result:
            for obj in pred:
                if obj[3] != -1:
                    result_np.append(list(obj[0]) + [obj[2]] + [obj[3]])
        result_np = np.array(result_np, dtype='uint8')
        #
        if len(result_np.shape) == 1:
            classes = np.array([])
            counts = np.array([])
        else:
            classes, counts = np.unique(result_np[:, 4], return_counts=True)
        print(result_np.shape, classes, counts)

        #
        classes = list(classes)
        counts = list(counts)
        for i in range(len(classes)):
            print(i, classes[i], counts[i])
            print(result_np[result_np[:, 4] == classes[i]])
            # x1 = result_np[result_np[:, 4] == classes[i]][:, 0]
            x = result_np[result_np[:, 4] == classes[i]][:, 0:4]
            print(x)
            print(np.mean(x, axis=0).round())


    # TODO: В дальнейшем результат = обработанный аккумулятор. Временно берем последний предикт
    result = acc_result[-1] if len(acc_result) > 0 else []

    for res in result:
        # print("res", res)
        (X1, Y1, X2, Y2), _, class_id, track_id = res
        # COLOR = colors[class_id]
        COLOR = colors[track_id]
        frame = cv.rectangle(frame, (X1, Y1), (X2, Y2), COLOR, thickness=2)
        frame = cv.putText(frame, names[class_id], (X1, Y1 + 10), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=COLOR, thickness=1)

    #
    cv.imshow(RTSP_URL, frame)
    #
    c = cv.waitKey(1)
    if c == 27:
        if DEBUG:
            print("Останавливаем thread и выходим из цикла получения и обработки фреймов")
        myThread.stop = True
        break

#
if DEBUG:
    print("Отключаем capture, закрываем все окна")
cap.release()
cv.destroyAllWindows()


