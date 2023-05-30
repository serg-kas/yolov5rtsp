import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import threading
import os
import time
from time import sleep
from random import randint
import urllib.request
import json
import subprocess

DEBUG = True
# RTSP_URL = 'rtsp://admin:daH_2019@192.168.5.44:554/cam/realmonitor?channel=13&subtype=0'
RTSP_URL = 'rtsp://admin:daH_2019@192.168.5.44:554/cam/realmonitor?channel=13&subtype=0'

# #############################################################################
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;udp"
cap = cv.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise IOError("Cannot open cam: {}".format(RTSP_URL))
else:
    if DEBUG:
        print("Подключили capture: {}".format(RTSP_URL))


def open_ffmpeg_stream_process():
    args = ("ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
            "rgb24 -s 1280x720 -i pipe:0 -pix_fmt yuv420p "
            "-f rtsp rtsp://rtsp_server:8554/stream"
            ).split()
    # return subprocess.Popen(args, stdin=subprocess.PIPE)
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def capture_loop():
    ffmpeg_process = open_ffmpeg_stream_process()
    # capture = cv.VideoCapture(RTSP_URL)
    while True:
        grabbed, frame = cap.read()
        # print(frame.shape)
        if not grabbed:
            break
        # cv.imshow("frame", frame)
        # ffmpeg_process.stdin.write(frame.astype(np.uint8).tobytes())
        # ffmpeg_process.stdin.write(frame.tobytes())
        output = ffmpeg_process.communicate(input=frame.tobytes())[0]
        # c = cv.waitKey(1)
        # if c == 27:
        #     break
    cap.release()
    cv.destroyAllWindows()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


capture_loop()
# "ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt rgb24 -s 1280x720 -i pipe:0 -pix_fmt yuv420p -f rtsp rtsp://rtsp_server:8554/stream"