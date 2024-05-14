FROM ubuntu:22.04

RUN apt update

RUN apt install -y python3

RUN apt install -y curl

RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"

RUN python3 get-pip.py

RUN pip3 install kserve torch torchvision

COPY model.py model.py

ENTRYPOINT python3 -m model