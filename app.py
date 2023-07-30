import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
# import os
import random
import threading
import time
# import json
# import urllib.request
# from urllib.parse import quote
#
import settings as s
import log
import tlg
import utils_small as u

# ########################## Получение параметров #############################
# Флаг вывода отладочных сообщений
DEBUG = s.DEBUG

logger = log.get_logger(level=log.DEBUG if s.DEBUG else log.INFO)
logger.info("Logger initialized")

# Показывать видео на экране
SHOW_VIDEO = s.SHOW_VIDEO

# Транслировать видео на rtsp сервер
VIDEO_TO_RTSP = s.VIDEO_TO_RTSP

# Целевая ширина фрейма для обработки и показа изображения
def_W = s.def_W

# Телеграм
bot_Token = s.bot_Token
chat_Id = s.chat_Id
url_tg = s.url_tg

logger.debug("Initialization Telegram sender")
tlg_sender = tlg.TlgSender(
    log_level=log.DEBUG if s.DEBUG else log.INFO,
    bot_token=s.bot_Token,
    chat_id=s.chat_Id
)

# RTSP для получения и трансляции видео
RTSP_URL = s.RTSP_URL
RTSP_SERVER = s.RTSP_SERVER


# #############################################################################
# Для трансляции видео должен быть предварительно запущен RTSP сервер
# rtsp://<IP>:8554/mystream
if VIDEO_TO_RTSP:
    #
    command_ffmpeg = ("ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
                      "rgb24 -s 800x450 -i pipe:0 -pix_fmt yuv420p -c:v libx264 "
                      "-f rtsp {}").format(RTSP_SERVER)
    #
    ffmpeg_process = u.open_ffmpeg_stream_process(command=command_ffmpeg.split(' '))

# #############################################################################
# Случайные цвета для треков
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for x in range(100)]

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

# Классы, которые детектим
names_to_detect = s.NAMES_TO_DETECT
# Если детектим всё, то names_to_detect = names
# names_to_detect = names

classes_list = []
for idx, name in enumerate(names):
    if name in names_to_detect:
        classes_list.append(idx)

logger.debug("Детектируем классы. Индексы: {}".format(classes_list))

# Паттерн (сочетание классов), который ищем
pattern_names = s.PATTERN_NAMES
#
pattern_list = []
for idx, name in enumerate(names):
    if name in pattern_names:
        pattern_list.append(idx)

logger.debug("Распознаем паттерн. Индексы: {}".format(pattern_list))


# #############################################################################
class MyThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.logger = log.get_logger(name="neural_net", level=log.DEBUG if s.DEBUG else log.INFO)
        #
        self.result = None          # результаты предикта
        self.image = None           # сюда подавать изображение для предикта
        self.stop = False           # остановить поток
        #
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        self.model.conf = s.MODEL_CONF  # confidence threshold (0-1)
        self.model.iou = s.MODEL_IOU    # NMS IoU threshold (0-1)
        self.model.classes = classes_list
        # HOOK: prevent error 'Upsample' object has no attribute 'recompute_scale_factor'
        for m in self.model.modules():
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None

    def run(self):
        self.logger.info("Start loop")
        while not self.stop:
            # подождать если изображения нет
            if self.image is None:
                self.logger.debug("Нейронка свободна, но фрейма для предикта нет")
                time.sleep(0.01)
                continue
            else:
                self.logger.debug("Есть фрейм, получаем предикт")
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
        self.logger.info("Stop loop")


# #############################################################################
# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;udp"
attempts = s.ATTEMPTS
logger.info("Connecting to the camera at %s", RTSP_URL)
_, cap = u.get_cap(RTSP_URL, max_attempts=attempts)
if cap is None:
    logger.error("Cannot open cam: {} after {} attempts".format(RTSP_URL, attempts))
    exit(1)
else:
    #
    logger.info("Camera connected successfully")

    # Getting frame for sending to Telegram
    ret, frame = cap.read()
    image = cv.imencode('.jpg', frame)
    try:
        tlg_sender.send(
            "Камера успешно подключена: {}".format(RTSP_URL),
            image[1].tobytes()
        )
    except tlg.QueueFull as error:
        logger.error(error)
    else:
        logger.info("Отправили сообщение в тлг о подключении камеры")

logger.info("Running Telegram sender")
# TODO: Почему-то надо запускать, когда в очередь уже что-то добавили (tlg_sender.send).
tlg_sender.start()

# #############################################################################
myThread = MyThread()
myThread.start()
#
result_show = None  # список объектов для отображения на фрейме
#
N_acc = s.N_ACC      # размер аккумулятора предиктов
N_pred = s.N_PRED    # при скольких предиктах объекта в аккумуляторе показываем объект
N_coord = s.N_COORD  # по скольки последним предиктам усредняем bb (координаты)
assert N_pred <= N_acc and N_coord < N_acc
#
acc_result_np = np.zeros((1, 7), dtype=np.int32)  # аккумулируемый результат предиктов numpy
acc_result_np[0, 5:7] = -1                              # фейковые номер объекта и трек -1
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
        logger.debug("Получен новый фрейм от источника")
        #
        h, w = frame.shape[:2]
        W = def_W
        H = int(W / w * h)
        frame = cv.resize(frame, (W, H), interpolation=cv.INTER_AREA)

        # засылаем новый фрейм на предикт
        if myThread.image is None:
            myThread.image = frame[:, :, ::-1].copy()
            logger.debug("Новый фрейм подан на предикт")
        else:
            logger.debug("Нейронка занята, фрейм не берет")
    else:
        cap_error_count += 1  # счетчик ошибок при получении изображения
        logger.debug("Нет нового фрейма от источника, cap_error_count: {}".format(cap_error_count))
        if cap_error_count > cap_errors_reconnect:
            del cap
            _, cap = u.get_cap(RTSP_URL, max_attempts=attempts)
            if cap is None:
                logger.error("Cannot reconnect cam: {} after {} attempts".format(RTSP_URL, attempts))
                exit(1)
        logger.debug("Переподключили источник capture")
    #
    if myThread.result is not None:
        # Получаем и обрабатываем результат предикта
        new_result_np = myThread.result.copy()
        myThread.result = None
        # print(new_result_np.shape, new_result_np)
        logger.debug("Получен результат numpy, размерности {}".format(new_result_np.shape))
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
                if new_obj == obj and u.get_iou(new_obj_coord, obj_coord) > s.IOU_to_track:
                    # print(u.get_iou(new_obj_coord, obj_coord))
                    logger.debug("  Объекту {} присвоен существующий трек {}".format(names[new_obj], obj_track))
                    new_result_np[i, 6] = obj_track
                    obj_recognized = True
                    break
                elif new_obj != obj and u.get_iou(new_obj_coord, obj_coord) > s.IOU_to_rename:
                    logger.debug("  Объект {} переименован в {}, сохранен существующий трек {}".format(names[obj],
                                                                                                       names[new_obj],
                                                                                                       obj_track))
                    new_result_np[i, 6] = obj_track
                    obj_recognized = True
                    break
            if obj_recognized:
                continue
            else:
                logger.debug("  Объекту {} присвоен новый трек {}".format(names[new_obj], track))
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
            color_id = track_id % len(colors)
            COLOR = colors[color_id]
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
                logger.debug("Pattern coordinates: %d, %d, %d, %d", X1, Y1, X2, Y2)
        logger.debug("Person coordinates: %d, %d, %d, %d", *person_coords)
        # print(pattern_coord_list)
        pattern_txt = ""
        pattern_show_coord = [-1, -1, -1, -1]
        if -1 not in person_coords and len(pattern_coord_list) > 0:
            pattern_coord_np = np.array(pattern_coord_list)
            # координаты person
            X1_person, Y1_person, X2_person, Y2_person = person_coords
            Xc_person = (X1_person + X2_person) / 2
            Yc_person = (Y1_person + Y2_person) / 2
            logger.debug("Person center coordinates: %d, %d", Xc_person, Yc_person)
            # координаты паттерна
            X1_pat = np.min(pattern_coord_np[:, 0])
            Y1_pat = np.min(pattern_coord_np[:, 1])
            X2_pat = np.max(pattern_coord_np[:, 2])
            Y2_pat = np.max(pattern_coord_np[:, 3])
            Xc_pat = (X1_pat + X2_pat) / 2
            Yc_pat = (Y1_pat + Y2_pat) / 2
            logger.debug("Coordinates of the center of combining patterns: %d, %d", Xc_pat, Yc_pat)
            # условие "близости" здесь центры ближе половины экрана
            X_distance = abs(Xc_person - Xc_pat)
            logger.debug("Distance between objects along the X axis: %d", X_distance)
            Y_distance = abs(Yc_person - Yc_pat)
            logger.debug("Distance between objects along the Y axis: %d", Y_distance)
            # if (X_distance < def_W / 2) and (Y_distance < def_W / 2):
            if (X_distance < def_W) and (Y_distance < def_W):
                logger.info("Паттерн найден: {}".format(pattern_names))
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
            diff_time = msg_time - start
            logger.debug("diff_time: %d", diff_time)
            if diff_time > 20:
                start = time.time()
                # Изображение в телегу
                # logger.info("Отправляем изображение в тлг старой функцией")
                # u.send_image_tlg(frame, bot_Token, chat_Id)

                logger.info("Отправляем изображение в тлг")
                image = cv.imencode('.jpg', frame)
                try:
                    tlg_sender.send(
                        "Сработал детектор",
                        image[1].tobytes()
                    )
                except tlg.QueueFull as error:
                    logger.error(error)
                else:
                    logger.info("Image sent")
    #
    if VIDEO_TO_RTSP:
        if frame is not None:
            logger.debug("Посылаем фрейм на rtsp сервер")
            #
            frame_rtsp = frame[:, :, ::-1]
            ffmpeg_process.stdin.write(frame_rtsp.astype(np.uint8).tobytes())
            #
            prev_frame_rtsp = frame_rtsp.copy()
            prev_frame_rtsp = cv.circle(prev_frame_rtsp, (30, 30), 10, u.red, -1)
        else:
            if prev_frame is not None:
                logger.debug("Фрейм для отправки на rtsp сервер is None. Посылаем предыдущий фрейм.")
                ffmpeg_process.stdin.write(prev_frame_rtsp.astype(np.uint8).tobytes())

    #
    if SHOW_VIDEO:
        if frame is not None:
            logger.debug("Выводим фрейм на экран")
            #
            cv.imshow(RTSP_URL, frame)
            #
            prev_frame = frame.copy()
            prev_frame = cv.circle(prev_frame, (30, 30), 10, u.red, -1)
        else:
            logger.debug("Фрейм для вывода на экран is None. Выводим предыдущий фрейм.")
            cv.imshow(RTSP_URL, prev_frame)
            #
        c = cv.waitKey(1)
        if c == 27:
            logger.debug("Останавливаем thread и выходим из цикла получения и обработки фреймов")
            myThread.stop = True
            break
#
logger.info("Stopping Telegram sender")
tlg_sender.stop()
#
logger.info("Отключаем capture, закрываем все окна")
cap.release()
cv.destroyAllWindows()
#
if VIDEO_TO_RTSP:
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
