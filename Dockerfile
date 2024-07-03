FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    dpkg \
    && rm -rf /var/lib/apt/lists/* \
    && wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb \
    && dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb \
    && rm libssl1.1_1.1.0g-2ubuntu4_amd64.deb

WORKDIR /app

COPY requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

WORKDIR /app

