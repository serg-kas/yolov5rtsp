FROM ubuntu:latest
#FROM python:3.10.5-slim-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

#EXPOSE 8554/tcp
#EXPOSE 8554/udp

CMD [ "python3", "app.py"]
