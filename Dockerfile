FROM ubuntu:18.04
MAINTAINER fnndsc "dev@babymri.org"

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration \
  && apt-get install -y python3-pip python3-dev nvidia-361-dev git \
  && ln -s /usr/bin/python3 /usr/local/bin/python \
  && python3 -m pip install --upgrade pip \
  && pip3 install setuptools --upgrade \
  && pip3 install pillow xlrd pydicom pandas wheel tensorflow==1.15 opencv-python==4.2.0.34 numpy matplotlib \
  && pip3 install -U scikit-learn \
  && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 \
  && mkdir -p /usr/local/cuda/lib \
  && cp /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib/ \
  && git clone https://github.com/lindawangg/COVID-Net.git

CMD ["/bin/bash"]
