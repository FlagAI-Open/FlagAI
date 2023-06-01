FROM ubuntu:23.10

WORKDIR app

ADD ./version.txt version

# sed -i s/ports.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list &&\
# sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list &&\
# sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list &&\
# pip install -i https://mirrors.aliyun.com/pypi/simple/ \

RUN \
apt-get update && \
apt-get install -y git pkg-config bash \
python3-full python3-pip python3-aiohttp &&\
update-alternatives --install /usr/bin/python python /usr/bin/python3 1 &&\
pip install --break-system-packages \
setuptools==66.0.0 \
urllib3==1.26.16 \
scipy transformers huggingface_hub packaging \
tqdm requests cython \
torch onnx && apt-get clean -y

ARG GIT
ENV GIT=$GIT

RUN git clone --depth=1 $GIT

ADD os/ /

RUN cd FlagAI &&\
  rm setup.cfg &&\
  python setup.py install

RUN pip uninstall -y google-auth 

ENV PYTHONPATH /app

