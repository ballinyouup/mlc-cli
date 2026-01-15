FROM ubuntu:24.04
LABEL authors="bryan"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    wget \
    git \
    git-lfs \
    cmake \
    build-essential \
    pkg-config \
    autoconf \
    automake \
    libtool \
    m4 \
    perl \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-dev \
    tar \
    xz-utils \
    unzip \
    zip \
    patch \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://go.dev/dl/go1.24.0.linux-amd64.tar.gz | tar -C /usr/local -xzf -
ENV PATH="/usr/local/go/bin:${PATH}"

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /usr/local/miniconda \
    && rm /tmp/miniconda.sh \
    && /usr/local/miniconda/bin/conda init bash \
    && /usr/local/miniconda/bin/conda config --set auto_activate_base false

ENV PATH="/usr/local/miniconda/bin:${PATH}"

WORKDIR /home/mlc-cli

COPY . .

RUN go mod download

ENTRYPOINT ["/bin/bash"]