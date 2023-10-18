FROM ubuntu:22.04

RUN apt update && apt-get install -y apt-transport-https ca-certificates curl gnupg

RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg| gpg -o /usr/share/keyrings/kubernetes-archive-keyring.gpg --dearmor

RUN echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] http://apt.kubernetes.io/ kubernetes-xenial main" | tee -a /etc/apt/sources.list

RUN apt update && apt install kubectl

RUN apt install jq -y