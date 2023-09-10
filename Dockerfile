# Stage 1: Builder/Compiler
FROM python:3.10.5-slim-buster as builder

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
	    build-essential software-properties-common && \
	add-apt-repository -y ppa:deadsnakes/ppa && \
	apt install --no-install-recommends --yes \
	    python3.10 python3-distutils \
	    ffmpeg libsm6 libxext6 \
        libfdk-aac-dev libass-dev libopus-dev libtheora-dev \
		libvorbis-dev libvpx-dev libssl-dev \
        curl && \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
	apt clean && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local/lib/python3.10/site-packages /usr/local/lib/python3.10/dist-packages

WORKDIR /app

COPY . .

CMD ["python3", "app.py"]
