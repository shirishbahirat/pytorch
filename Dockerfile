FROM ubuntu:latest

MAINTAINER sbh

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python3-opengl \
       python3-pip \
       python3-dev

RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/

RUN pip3 install --trusted-host pypi.python.org -r /tmp/requirements.txt

RUN rm /tmp/requirements.txt

RUN mkdir /home/pytorch/

WORKDIR /home/pytorch/

CMD bash
