FROM ubuntu:latest

COPY mnist_model /mnist_model

COPY script.sh /script.sh

RUN chmod +x /script.sh

ENTRYPOINT [ "/script.sh" ]



