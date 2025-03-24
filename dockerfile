FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libatlas-base-dev \
    libboost-all-dev \
    libarpack++2-dev \
    cmake \
    git \
    python3 \
    texlive-full \
    biber \
    latexmk \
    make \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    pip install numpy matplotlib 

WORKDIR /home

RUN git clone https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX.git && \
    cd tRecX && \
    cmake . && \
    make -j6

RUN git clone https://github.com/joporsch04/Bachelor-thesis.git /home/Bachelor-thesis

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

CMD ["/bin/bash"]