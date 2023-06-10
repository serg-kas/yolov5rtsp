import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import os
import random
import threading
import time
import json
import urllib.request
#
import utils_small as u

# Флаг вывода отладочных сообщений
DEBUG = False

# #############################################################################
def_W = 800           # целевая ширина фрейма для обработки и показа изображения
#
SHOW_VIDEO = False    # показывать видео на экране
#
VIDEO_to_RTSP = False  # транслировать видео на rtsp сервер

# #############################################################################
bot_Token = "6260918240:AAFSXBtd5gHJHdrgbyKoDsJkZYO1E9SSHUs"
# url_tg_admin = "https://api.telegram.org/bot" + bot_Token + "/sendMessage?chat_id=" + chat_Id_admin + "&text=TEXT!"
# url_tg = "https://api.telegram.org/bot" + bot_Token + "/sendMessage?chat_id=" + chat_Id + "&text=TEXT!"

# #############################################################################
url_json = "https://modulemarket.ru/api/22ac5704-dfc3-11ed-b813-000c29be8d8a/getparams?appid=5"
try:
    url = urllib.request.urlopen(url_json)
    data = json.load(url)
    RTSP_URL = data[0]['in']['url']
    chat_Id_admin = data[1]['out']['telegram']
    chat_Id = data[0]['out']['telegram']
    print("Получены настройки {}".format([RTSP_URL, chat_Id_admin, chat_Id]))

except urllib.error.URLError as e:
    print("Ошибка получения параметров из json: {}".format(e.reason))
    RTSP_URL = "rtsp://admin:12345678q@212.45.21.150/cam/realmonitor?channel=1&subtype=1"
    chat_Id_admin = "47989888"
    chat_Id = "1443607497"
    print("Дефолтные настройки {}".format([RTSP_URL, chat_Id_admin, chat_Id]))


# TODO: переназначаем для тестов на дом.камеру
# RTSP_URL = 'rtsp://admin:daH_2019@192.168.5.44:554/cam/realmonitor?channel=13&subtype=0'
# RTSP_server = 'rtsp://localhost:8554/mystream'
RTSP_server = 'rtsp://192.168.5.151:8554/mystream'  # для запуска из докера

# #############################################################################
if VIDEO_to_RTSP:
    # RSTP сервер mediamtx должен быть предварительно запущен
    # rtsp://<IP>:8554/mystream
    command_ffmpeg = ("ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
                      "rgb24 -s 800x450 -i pipe:0 -pix_fmt yuv420p -c:v libx264 "
                      "-f rtsp {}").format(RTSP_server)

    ffmpeg_process = u.open_ffmpeg_stream_process(command=command_ffmpeg.split(' '))

# #############################################################################
# Случайные цвета для треков
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for x in range(1000)]

# Начальный номер трека
track = 1

# Имена классов
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

# Классы, которые детектим (если все, то = names)
names_to_detect = ['person', 'laptop', 'bottle']
# names_to_detect = names
#
classes_list = []
for idx, name in enumerate(names):
    if name in names_to_detect:
        classes_list.append(idx)
if DEBUG:
    print("Детектируем классы. Индексы: {}".format(classes_list))

# Паттерн (сочетание классов), который ищем
pattern_names = ['person', 'bottle']
#
pattern_list = []
for idx, name in enumerate(names):
    if name in pattern_names:
        pattern_list.append(idx)
if DEBUG:
    print("Распознаем паттерн. Индексы: {}".format(pattern_list))


# #############################################################################
class MyThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None          # результаты предикта
        self.image = None           # сюда подавать изображение для предикта
        self.stop = False           # остановить поток
        #
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        self.model.conf = 0.25  # confidence threshold (0-1)
        self.model.iou = 0.45   # NMS IoU threshold (0-1)
        self.model.classes = classes_list
        # HOOK: prevent error 'Upsample' object has no attribute 'recompute_scale_factor'
        for m in self.model.modules():
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None

    def run(self):
        while not self.stop:
            # подождать если изображения нет
            if self.image is None:
                if DEBUG:
                    print("MyThread: Нейронка свободна, но фрейма для предикта нет")
                time.sleep(0.01)
                continue
            else:
                if DEBUG:
                    print("MyThread: Есть фрейм, получаем предикт")
                # получаем предикт
                results = self.model([self.image]).xyxy[0]
                # формируем результаты в массив numpy
                result_numpy = results.numpy()
                track_numpy = np.full((result_numpy.shape[0], 1), -1)  # -1 номер трека по умолчанию
                result_numpy = np.append(result_numpy, track_numpy, axis=1)
                # print(result_numpy.shape, result_numpy)
                #
                self.result = result_numpy
                self.image = None


# #############################################################################
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;udp"
attempts = 3
_, cap = u.get_cap(RTSP_URL, max_attempts=attempts)
if cap is None:
    print("Cannot open cam: {} after {} attempts".format(RTSP_URL, attempts))
    exit(1)

# #############################################################################
myThread = MyThread()
myThread.start()
#
result_show = None  # список объектов для отображения на фрейме
#
N_acc = 50     # размер аккумулятора, предиктов
N_pred = 2     # при скольких предиктах объекта в аккумуляторе показываем объект
N_coord = 5    # по скольки последним предиктам усредняем bb (координаты)
assert N_pred <= N_acc and N_coord < N_acc
#
acc_result_np = np.zeros((1, 7), dtype=np.int32)  # аккумулируемый результат предиктов numpy
acc_result_np[0, 5:7] = -1                        # фейковые номер объекта и трек -1
#
# Предыдущий фрейм если не готов новый
prev_frame_rtsp = None
prev_frame = None
#
start = time.time()  # Начало засечки времени
#
# После скольки ошибок чтения переподключить источник
cap_errors_reconnect = 5
cap_error_count = 0

# #############################################################################
while True:
    # получаем новый фрейм
    ret, frame = cap.read()
    if ret:
        cap_error_count = 0  # обнуляем счетчик ошибок при получении изображения
        if DEBUG:
            print("Получен новый фрейм от источника")
        #
        h, w = frame.shape[:2]
        W = def_W
        H = int(W / w * h)
        frame = cv.resize(frame, (W, H), interpolation=cv.INTER_AREA)

        # засылаем новый фрейм на предикт
        if myThread.image is None:
            myThread.image = frame[:, :, ::-1].copy()
            if DEBUG:
                print("Новый фрейм подан на предикт")
        else:
            if DEBUG:
                print("Нейронка занята, фрейм не берет")
    else:
        cap_error_count += 1  # счетчик ошибок при получении изображения
        if DEBUG:
            print("Нет нового фрейма от источника, cap_error_count: {}".format(cap_error_count))
        if cap_error_count > cap_errors_reconnect:
            del cap
            _, cap = u.get_cap(RTSP_URL, max_attempts=attempts)
            if cap is None:
                print("Cannot reconnect cam: {} after {} attempts".format(RTSP_URL, attempts))
                exit(1)
        if DEBUG:
            print("Переподключили источник capture")
    #
    if myThread.result is not None:
        # Получаем и обрабатываем результат предикта
        new_result_np = myThread.result
        # print(new_result_np.shape, new_result_np)
        if DEBUG:
            print("Получен результат numpy, размерности {}".format(myThread.result.shape))
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
                if new_obj == obj and u.get_iou(new_obj_coord, obj_coord) > 0.40:
                    # print(u.get_iou(new_obj_coord, obj_coord))
                    if DEBUG:
                        print("  Объекту {} присвоен существующий трек {}".format(names[new_obj], obj_track))
                    new_result_np[i, 6] = obj_track
                    obj_recognized = True
                    break
                elif new_obj != obj and u.get_iou(new_obj_coord, obj_coord) > 0.90:
                    if DEBUG:
                        print("  Объект {} переименован в {}, сохранен существующий трек {}".format(names[obj],
                                                                                                  names[new_obj],
                                                                                                  obj_track))
                    new_result_np[i, 6] = obj_track
                    obj_recognized = True
                    break
            if obj_recognized:
                continue
            else:
                if DEBUG:
                    print("  Объекту {} присвоен новый трек {}".format(names[new_obj], track))
                new_result_np[i, 6] = track
                track += 1 if track < 1000 else 0
        #
        new_result_to_add = new_result_np[new_result_np[:, 6] != -1]
        acc_result_np = np.append(acc_result_np, new_result_to_add, axis=0)

        if acc_result_np.shape[0] > N_acc:
            acc_result_np = acc_result_np[-N_acc:, :]
        # print(acc_result_np.shape, acc_result_np)

        # Получаем уникальные номера треков и их количество в аккумуляторе
        tracks, counts = np.unique(acc_result_np[:, 6], return_counts=True)
        # print("tracks", tracks, "counts", counts)

        result_show = []
        for i in range(len(tracks)):
            # print("tracks[i]", tracks[i], "counts[i]", counts[i])
            if counts[i] > N_pred:
                track_np = acc_result_np[acc_result_np[:, 6] == tracks[i]]
                coords_np = track_np[:, 0:4]
                if coords_np.shape[0] > N_coord:
                    coords_np = coords_np[-N_coord:, :]
                coords_mean = tuple(np.mean(coords_np, axis=0).astype(int))
                #
                result_show.append([coords_mean, 1, int(track_np[0, 5]), int(tracks[i])])
                # print("result_show[-1]", result_show[-1])
    #
    if result_show is not None:
        for res in result_show:
            (X1, Y1, X2, Y2), _, class_id, track_id = res
            # print((X1, Y1, X2, Y2), class_id, track_id)
            #
            COLOR = colors[track_id]
            #
            frame = cv.rectangle(frame, (X1, Y1), (X2, Y2), COLOR, thickness=2)
            frame = cv.putText(frame, names[class_id], (X1 + 5, Y1 + 10),
                               cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=COLOR, thickness=1)
            frame = cv.putText(frame, 'id ' + str(track_id), (X1 + 5, Y1 + 20),
                               cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=COLOR, thickness=1)

        # Алгоритм поиска паттерна - близость центра person и усредненного центра остальных объектов
        person_coords = [-1, -1, -1, -1]
        pattern_coord_list = []
        #
        for res in result_show:
            (X1, Y1, X2, Y2), _, class_id, track_id = res
            if class_id == 0:
                person_coords = [X1, Y1, X2, Y2]
                continue
            if class_id in pattern_list and class_id != 0:
                pattern_coord_list.append([X1, Y1, X2, Y2])
        # print(pattern_coord_list)
        pattern_txt = ""
        pattern_show_coord = [-1, -1, -1, -1]
        if -1 not in person_coords and len(pattern_coord_list) > 0:
            pattern_coord_np = np.array(pattern_coord_list)
            # координаты person
            X1_person, Y1_person, X2_person, Y2_person = person_coords
            Xc_person = (X1_person + X2_person) / 2
            Yc_person = (Y1_person + Y2_person) / 2
            # координаты паттерна
            X1_pat = np.min(pattern_coord_np[:, 0])
            Y1_pat = np.min(pattern_coord_np[:, 1])
            X2_pat = np.max(pattern_coord_np[:, 2])
            Y2_pat = np.max(pattern_coord_np[:, 3])
            Xc_pat = (X1_pat + X2_pat) / 2
            Yc_pat = (Y1_pat + Y2_pat) / 2
            # условие "близости" здесь центры ближе половины экрана
            if (abs(Xc_person - X1_pat) < def_W / 2) and (abs(Yc_person - Yc_pat) < def_W / 2):
                if DEBUG:
                    print("Паттерн найден: {}".format(pattern_names))
                pattern_txt = "Attention: {}".format(pattern_names)
                X1_show = min(X1_person, X1_pat)
                Y1_show = min(Y1_person, Y1_pat)
                X2_show = max(X2_person, X2_pat)
                Y2_show = max(Y2_person, Y2_pat)
                pattern_show_coord = [X1_show, Y1_show, X2_show, Y2_show]
        #
        if len(pattern_txt) > 0:
            X1, Y1, X2, Y2 = pattern_show_coord
            frame = cv.rectangle(frame, (X1, Y1), (X2, Y2), u.red, thickness=2)
            frame = cv.putText(frame, pattern_txt, (X1, Y1 + 15),
                               cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=u.red, thickness=2)
            #
            msg_time = time.time()
            if msg_time - start > 20:
                start = time.time()

                # Сообщение в телегу
                # if DEBUG:
                #     print("Отправляем сообщение в тлг")
                # with urllib.request.urlopen(url_tg) as response:
                #     html = response.read()
                #     print(html)

                # Изображение в телегу
                if DEBUG:
                    print("Отправляем изображение в тлг")
                # u.send_image_tlg(frame, bot_Token, chat_Id_admin)
                u.send_image_tlg(frame, bot_Token, chat_Id)
    #
    if VIDEO_to_RTSP:
        if frame is not None:
            if DEBUG:
                print("Посылаем фрейм на rtsp сервер")
            #
            frame_rtsp = frame[:, :, ::-1]
            ffmpeg_process.stdin.write(frame_rtsp.astype(np.uint8).tobytes())
            #
            prev_frame_rtsp = frame_rtsp.copy()
            prev_frame_rtsp = cv.circle(prev_frame_rtsp, (30, 30), 10, u.red, -1)
        else:
            if DEBUG:
                print("Фрейм для отправки на rtsp сервер is None. Посылаем предыдущий фрейм.")
            ffmpeg_process.stdin.write(prev_frame_rtsp.astype(np.uint8).tobytes())

    #
    if SHOW_VIDEO:
        if frame is not None:
            if DEBUG:
                print("Выводим фрейм на экран")
            #
            cv.imshow(RTSP_URL, frame)
            #
            prev_frame = frame.copy()
            prev_frame = cv.circle(prev_frame, (30, 30), 10, u.red, -1)
        else:
            if DEBUG:
                print("Фрейм для вывода на экран is None. Выводим предыдущий фрейм.")
            cv.imshow(RTSP_URL, prev_frame)
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
#
if VIDEO_to_RTSP:
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


