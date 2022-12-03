FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt-get update -y
RUN apt-get install -y python3.10.4
RUN apt-get -y install pip

COPY /src /src
COPY .env .
COPY requirements.txt .

RUN pip install -r requirements.txt