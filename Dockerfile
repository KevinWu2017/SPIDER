FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install base tooling and Python venv support.
RUN apt-get update \
	&& apt-get install -y build-essential python3 python3-venv \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /SPIDER

COPY . /SPIDER
