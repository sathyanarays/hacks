FROM ubuntu:22.04

RUN apt update

COPY mpich_4.0-3_amd64.deb .

RUN sh -c echo Hello

RUN apt install -y ./mpich_4.0-3_amd64.deb

COPY helloworld /helloworld

WORKDIR /helloworld

RUN apt install -y build-essential openssh-client openssh-server

RUN make mpi_hello_world

COPY send_recv /send_recv

WORKDIR /send_recv

RUN make mpi_send_recv

COPY pingpong /pingpong

WORKDIR /pingpong

RUN make pingpong

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

RUN echo "root:pass1234" | chpasswd