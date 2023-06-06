FROM ubuntu:latest
#FROM python:3.10.5-slim-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libfdk-aac-dev libass-dev libopus-dev libtheora-dev libvorbis-dev libvpx-dev libssl-dev -y
RUN apt-get install python3-pip -y

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install psutil
RUN pip3 install gitpython
RUN pip3 install ultralytics


COPY . .

#EXPOSE 8554/tcp
#EXPOSE 8554/udp

CMD [ "python3", "app.py"]
