# TODO: rearrange installs for docker build optimization
FROM debian:bullseye-20240513-slim
RUN apt update
RUN apt install -y \
  make=4.3-4.1 \
  gcc=4:10.2.1-1 \
  python3=3.9.2-3 \
  python3-pip=20.3.4-4+deb11u1 \
  libcurl4=7.74.0-1.3+deb11u11 \
  curl=7.74.0-1.3+deb11u11
RUN pip3 install numpy==1.26.4
COPY ./src/glove_src/ /opt/glove/
WORKDIR /opt/glove/
RUN make
RUN mkdir -p /veld/code/
WORKDIR /veld/code/
RUN pip3 install notebook==7.2.1

