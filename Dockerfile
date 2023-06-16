FROM balenalib/rpi-debian:latest

MAINTAINER Ting

WORKDIR /docker_demo

COPY torch-1.13.0a0+git7c98e70-cp39-cp39-linux_aarch64.whl .

RUN apt-get update && apt-get install -y libgl1 libpq-dev build-essential python3-pip python3-opencv
RUN ln -s /opt/conda/bin/python3 /bin/python3
RUN pip install -r requirements.txt && \
    pip install torch-1.13.0a0+git7c98e70-cp39-cp39-linux_aarch64.whl
