FROM nvidia/cuda:10.2-devel
WORKDIR /build
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel
RUN pip3 install torch==1.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt install iputils-ping opensm libopensm-dev libibverbs1 libibverbs-dev -y --no-install-recommends
ENV TORCH_CUDA_ARCH_LIST=6.1;7.0;7.5
ENV BMT_AVX512=1
ADD other_requirements.txt other_requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r other_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ADD . .
RUN python3 setup.py install

WORKDIR /root
ADD example example