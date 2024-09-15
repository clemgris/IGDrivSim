FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get install -y cmake && \
    apt-get install -y ffmpeg && apt-get install -y tmux

RUN apt-get -y update && \
    apt-get -y install git vim sudo curl wget apt-transport-https ca-certificates gnupg \
                       libgl1 libglib2.0-0 \
                       python3.9 python3-distutils python3-pip python3-apt && \
    rm -rf /var/lib/apt/lists/*

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

ARG USERID
ARG USERNAME
ARG SERVERNAME
RUN useradd -u $USERID --create-home $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    adduser $USERNAME sudo

# Create the 'docker' group (check your status on the servers)
RUN groupadd -g 999 docker

# Add the user to the 'docker' group
RUN usermod -aG docker $USERNAME

USER $USERNAME

RUN python3 -m pip install --upgrade pip

ADD requirements.txt .
RUN PATH="${HOME}/.local/bin:${PATH}" python3 -m pip install -r requirements.txt
RUN PATH="${HOME}/.local/bin:${PATH}" python3 -m pip install --upgrade "jax[cuda12_pip]" \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Adding tmux configuration
ADD .tmux.conf .
RUN cp .tmux.conf $HOME/.tmux.conf

WORKDIR /data/$SERVERNAME/$USERNAME