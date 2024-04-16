# Use the official image as a parent image
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04


# Set the working directory in the container
WORKDIR /
ENV DEBIAN_FRONTEND=noninteractive


# Update the system and install basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    gcc-9 \
    g++-9 \
    python3-pip 

# RUN pip install torch torchvision torchaudio
RUN git clone https://github.com/OpenAccess-AI-Collective/axolotl
RUN cd /axolotl && pip3 install packaging && pip3 install -e '.[flash-attn,deepspeed]'

# Run requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt
