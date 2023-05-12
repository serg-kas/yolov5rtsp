import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import threading
import os
from time import sleep
from random import randint
#
import utils_small as u

#
RTSP_URL = 'rtsp://admin:daH_2019@192.168.5.44:554/cam/realmonitor?channel=7&subtype=0'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;udp"
#
DEBUG = False        # флаг отладочных сообщений
RES_to_list = False  # сохранять и обрабатывать результаты предикта в списке
def_W = 800          # целевая ширина фрейма для отображения
#
colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for x in range(1000)]  # случайные цвета по числу треков
track = 1            # начальный номер трека
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
# names_to_detect = ['person', 'bottle']
classes_list = []
for idx, name in enumerate(names):
    if name in names_to_detect:
        classes_list.append(idx)
if DEBUG:
    print("Детектим классов: {}".format(len(classes_list)))

#
pattern_base_name = 'toilet'
pattern_base = names.index(pattern_base_name)
#
patterns_to_recognize = ['toilet']
assert len(patterns_to_recognize) > 0
patterns = []
for idx, name in enumerate(names):
    if name in patterns_to_recognize:
        patterns.append(idx)
if DEBUG:
    print("Распознаем базовый паттерн {} и остальные: {}".format(pattern_base_name, patterns_to_recognize))


# Класс потока, в котором запускаем НС
class MyThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None          # результаты предикта
        self.image = None           # сюда подавать изображение для предикта
        self.stop = False           # остановить поток
        #
        self.result_to_list = RES_to_list
        #
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

        # self.model.conf = 0.25  # confidence threshold (0-1)
        # self.model.iou = 0.45  # NMS IoU threshold (0-1)

        # HOOK: prevent error 'Upsample' object has no attribute 'recompute_scale_factor'
        for m in self.model.modules():
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None

    def run(self):
        while not self.stop:
            # подождать если изображения нет
            if self.image is None:
                if DEBUG:
                    print("Нейронка свободна, но фрейма для предикта нет")
                sleep(0.01)  # пробовал 0.5, 1
                continue
            #
            results = self.model([self.image]).xyxy[0]
            # results = model([self.image]).xyxy[0]

            if self.result_to_list:
                # Формируем результаты в список
                result_list = []
                for row in results:
                    coords = tuple(row.int().numpy()[:-2])
                    conf = float(row[-2])
                    curr_class = int(row[-1])
                    if curr_class in classes_list:
                        result_list.append([coords, conf, curr_class, -1])  # -1 - номер трека по умолчанию
                #
                self.result = result_list
                self.image = None
            else:
                # Формируем результаты в массив numpy
                result_numpy = results.numpy()
                track_numpy = np.full((result_numpy.shape[0], 1), -1)  # -1 - номер трека по умолчанию
                result_numpy = np.append(result_numpy, track_numpy, axis=1)
                #
                self.result = result_numpy
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
N = 60  # размер аккумулятора, предиктов
n = 30  # сколько детекций объекта в аккумуляторе считаем достоверной детекцией
assert n <= N

#
acc_result = [] if RES_to_list else None          # аккумулируемый результат предиктов
result_show = [] if RES_to_list else None         # список объектов для  отображения на фрейме
#
acc_result_np = np.zeros((1, 7), dtype=np.int32)  # аккумулируемый результат предиктов numpy
acc_result_np[0, 5:7] = -1                        # фейковые номер объекта и трек
#
while True:
    # получаем новый фрейм
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            if frame.shape[0] > 200 and frame.shape[1] > 200:
                if DEBUG:
                    print("Получен новый фрейм")
                #
                h, w = frame.shape[:2]
                W = def_W
                H = int(W / w * h)
                frame = cv.resize(frame, (W, H), interpolation=cv.INTER_AREA)

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
            if RES_to_list:
                print("Получен предикт, классов:", len(myThread.result))
            else:
                print("Получен предикт, классов:", myThread.result.shape[0])

        # Обрабатываем список результатов
        if RES_to_list:
            result = []
            for new_obj in myThread.result:
                #
                obj_recognized = False
                for pred in acc_result:
                    for obj in pred:
                        #
                        if new_obj[2] == obj[2] and u.get_iou(new_obj[0], obj[0]) > 0.25:
                            if DEBUG:
                                print("Объекту {} присвоен существующий трек {}".format(names[obj[2]], obj[3]))
                            new_obj[3] = obj[3]
                            obj_recognized = True
                            break
                        elif new_obj[2] != obj[2] and u.get_iou(new_obj[0], obj[0]) > 0.95:
                            if DEBUG:
                                print("Объект {} переименован в {}, сохранен существующий трек {}".format(names[obj[2]],
                                                                                                          names[new_obj[2]],
                                                                                                          obj[3]))
                            new_obj[3] = obj[3]
                            obj_recognized = True
                            break
                    if obj_recognized:
                        break
                #
                if not obj_recognized:
                    if DEBUG:
                        print("Объекту {} присвоен новый трек {}".format(names[new_obj[2]], track))
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
            result_np = np.array(result_np)
            # print(result_np)
            #
            if len(result_np.shape) == 1:
                tracks, counts = np.array([]), np.array([])
            else:
                tracks, counts = np.unique(result_np[:, 5], return_counts=True)
            # print("tracks", tracks, "counts", counts)
            #
            tracks = list(tracks)
            counts = list(counts)
            result_show = []
            for i in range(len(tracks)):
                # print("tracks[i]", tracks[i], "counts[i]", counts[i])
                if counts[i] > n:
                    track_np = result_np[result_np[:, 5] == tracks[i]]
                    coords_np = track_np[:, 0:4]
                    coords_mean = tuple(np.mean(coords_np, axis=0).astype(int))
                    #
                    result_show.append([coords_mean, 1, track_np[0, 4], tracks[i]])
                    # print("result_show[-1]", result_show[-1])

        # Обрабатываем numpy
        else:
            if DEBUG:
                print("Получен результат numpy, размерности {}".format(myThread.result.shape))
            #
            new_result_np = myThread.result
            #
            for i in range(new_result_np.shape[0]):
                new_obj_coord = list(new_result_np[i, 0:4].astype(np.int32))
                new_obj = int(new_result_np[i, 5])
                # print(new_obj_coord, new_obj)
                #
                obj_recognized = False
                for k in range(acc_result_np.shape[0]):
                    obj_coord = list(acc_result_np[k, 0:4])
                    obj = int(acc_result_np[k, 5])
                    obj_track = int(acc_result_np[k, 6])
                    # print(obj_coord, obj, obj_track)
                    #
                    if new_obj == obj and u.get_iou(new_obj_coord, obj_coord) > 0.25:
                        # print(u.get_iou(new_obj_coord, obj_coord))
                        if DEBUG:
                            print("Объекту {} присвоен существующий трек {}".format(names[new_obj], obj_track))
                        new_result_np[i, 6] = obj_track
                        obj_recognized = True
                        break
                    elif new_obj != obj and u.get_iou(new_obj_coord, obj_coord) > 0.90:
                        if DEBUG:
                            print("Объект {} переименован в {}, сохранен существующий трек {}".format(names[obj],
                                                                                                      names[new_obj],
                                                                                                      obj_track))
                        new_result_np[i, 6] = obj_track

                        obj_recognized = True
                        break
                if obj_recognized:
                    break
                    #
                if not obj_recognized:
                    if DEBUG:
                        print("Объекту {} присвоен новый трек {}".format(names[new_obj], track))
                    new_result_np[i, 6] = track
                    track += 1 if track < 1000 else 0
            #
            new_result_to_add = new_result_np[new_result_np[:, 6] != -1]
            acc_result_np = np.append(acc_result_np, new_result_to_add, axis=0)
            if acc_result_np.shape[0] > N:
                acc_result_np = acc_result_np[-N:, :]
            # print(acc_result_np.shape, acc_result_np)

            tracks, counts = np.unique(acc_result_np[:, 6], return_counts=True)
            # print("tracks", tracks, "counts", counts)

            result_show = []
            for i in range(len(tracks)):
                # print("tracks[i]", tracks[i], "counts[i]", counts[i])
                if counts[i] > n:
                    track_np = acc_result_np[acc_result_np[:, 6] == tracks[i]]
                    coords_np = track_np[:, 0:4]
                    coords_mean = tuple(np.mean(coords_np, axis=0).astype(int))
                    #
                    result_show.append([coords_mean, 1, int(track_np[0, 5]), int(tracks[i])])
                    # print("result_show[-1]", result_show[-1])

    # TODO: алгоритм оптимизировать и переделать в numpy
    base_pattern_list = []
    patterns_found_list = []
    if result_show is not None:
        for res in result_show:
            (X1, Y1, X2, Y2), _, class_id, track_id = res
            if class_id == pattern_base:
                base_pattern_list.append(res)
            # elif class_id in patterns:
            if class_id in patterns:
                patterns_found_list.append(res)
    # print(base_pattern_list)
    # print(patterns_found_list)
    pattern_txt_list = []
    for base in base_pattern_list:
        (X1, Y1, X2, Y2), _, class_id, track_id = base
        Xc = (X1 + X2) / 2
        Yc = (Y1 + Y2) / 2
        for pat in patterns_found_list:
            (X1pat, Y1pat, X2pat, Y2pat), _, class_id_pat, track_id_pat = pat
            Xc_pat = (X1pat + X2pat) / 2
            Yc_pat = (Y1pat + Y2pat) / 2
            if (abs(Xc - Xc_pat) < def_W / 10) and (abs(Yc - Yc_pat) < def_W / 10):
                if DEBUG:
                    print("Паттерн найден: {} with {}".format(names[class_id], names[class_id_pat]))
                pattern_txt_list.append("Attention, found: {} with {}".format(names[class_id], names[class_id_pat]))
    #
    if len(pattern_txt_list) > 0:
        print(pattern_txt_list)
        # TODO: послать сообщение в телегу


    #
    if result_show is not None:
        for res in result_show:
            (X1, Y1, X2, Y2), _, class_id, track_id = res
            # print((X1, Y1, X2, Y2), class_id, track_id)
            #
            COLOR = colors[track_id]
            #
            frame = cv.rectangle(frame, (X1, Y1), (X2, Y2), COLOR, thickness=2)
            frame = cv.putText(frame, names[class_id], (X1 + 5, Y1 + 10), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=COLOR, thickness=1)
            frame = cv.putText(frame, 'id ' + str(track_id), (X1 + 5, Y1 + 20), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=COLOR, thickness=1)
        #
        for pattern_txt in pattern_txt_list:
            # print(pattern_txt)
            frame = cv.putText(frame, pattern_txt, (30, 30), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=u.red, thickness=2)

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


