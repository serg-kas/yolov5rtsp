FROM python:3.10.5-slim-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install gitpython
RUN pip3 install ultralytics
RUN pip3 install psutil

COPY . .

EXPOSE 8554/tcp
EXPOSE 8554/udp

#CMD [ "./mediamtx"]
#CMD [ "python3", "app.py"]
ENTRYPOINT ["/bin/sh"]
CMD ["./start.sh"]