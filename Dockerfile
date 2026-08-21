FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu20.04

LABEL zhanglu0704

ENV TZ=Asia/Shanghai

VOLUME /etc/localtime

ENV WORK_DID=/workspace

WORKDIR ${WORK_DID}

RUN apt update && \
    apt install -y g++ gcc cmake curl wget vim unzip git openssh-server net-tools python3-packaging && \
    apt install -y python3.9 python3.9-dev python3-pip && \
    apt clean -y && \
    rm -rf /var/cache/apt/archives 

RUN rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cu117

COPY requirements.txt  ${WORK_DID}/

RUN python -m pip install -r ${WORK_DID}/requirements.txt

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    # git checkout -f 23.05 && \
    # pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" . && \
    cd ../ && rm -rf apex

RUN git clone https://github.com/OpenBMB/BMTrain && \
    cd BMTrain && \
    git checkout -f 0.2.2 && \
    # python setup.py install --prefix=/usr/local/
    pip install -v . && \
    cd ../ && rm -rf BMTrain

RUN git clone https://github.com/FlagAI-Open/FlagAI.git && \
    cd FlagAI && \
    pip install -v . && \
    cd ../ && rm -rf FlagAI

RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config

RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -N "" && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    chmod og-wx /root/.ssh/authorized_keys

CMD service ssh start && tail -f /dev/null

# sudo docker build -f Dockerfile --shm-size='120g' -t flagai:dev-ubuntu20-cuda11.7-py39 .
