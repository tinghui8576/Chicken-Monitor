FROM debian:bullseye

MAINTAINER Ting

WORKDIR /docker_demo

# COPY torch-1.13.0a0+git7c98e70-cp39-cp39-linux_aarch64.whl .

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get -y install build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \ 
        libnss3-dev \
        libssl-dev \
        libgl1 \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
        ibglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        wget \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get purge -y imagemagick imagemagick-6-common 

RUN wget https://www.python.org/ftp/python/3.9.2/Python-3.9.2.tgz \
    && tar -xzf Python-3.9.2.tgz \
    && cd Python-3.9.2 \
    && ./configure --enable-optimizations \
    && make altinstall
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.9 1

RUN apt-get install -y python3-pip
#  python3-opencv  \
# python3.9  
# libatlas-base-dev libgl1 libpq-dev build-essential  
# RUN ln -s /opt/conda/bin/python3 /bin/python3
# RUN pip install -r requirements.txt && \
#     pip install torch-1.13.0a0+git7c98e70-cp39-cp39-linux_aarch64.whl
