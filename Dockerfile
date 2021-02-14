FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

# Install system packages
RUN apt-get update --fix-missing && \
  apt-get install -y \
  wget \
  libgtk2.0-dev \
  bzip2 \
  ca-certificates \
  curl \
  git \
  vim \
  g++ \
  gcc \
  graphviz \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libglib2.0-0 \
  libgl1-mesa-glx \
  libhdf5-dev \
  openmpi-bin \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /tf
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r /tf/requirements.txt

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
