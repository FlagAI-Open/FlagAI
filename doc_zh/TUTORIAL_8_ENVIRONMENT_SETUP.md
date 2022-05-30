# 多机训练模型搭建环境

- [多机训练模型搭建环境](#多机训练模型搭建环境)
- [一.  Docker](#一--docker)
  - [1.安装docker](#1安装docker)
  - [2.Docker 换源](#2docker-换源)
  - [3.安装显卡驱动（如已装可跳过）](#3安装显卡驱动如已装可跳过)
  - [4.配置nvidia-docker源：](#4配置nvidia-docker源)
  - [5.制作dockerfile](#5制作dockerfile)
    - [a.拉取nvidia 基础镜像, 创建临时文件夹（容器内，镜像创建完成后，删除）](#a拉取nvidia-基础镜像-创建临时文件夹容器内镜像创建完成后删除)
    - [b.配置apt 安装源,并安装一些linux 系统常用基础包](#b配置apt-安装源并安装一些linux-系统常用基础包)
    - [c.  安装最新版git(创建镜像clone 安装包)](#c--安装最新版git创建镜像clone-安装包)
    - [d. 安装  Mellanox OFED, 由于网络问题，推荐安装包下到本地后，再执行dockerfile](#d-安装--mellanox-ofed-由于网络问题推荐安装包下到本地后再执行dockerfile)
    - [e. 安装 nv_peer_mem](#e-安装-nv_peer_mem)
    - [f. 安装openmpi, 需先安装libevent 依赖包](#f-安装openmpi-需先安装libevent-依赖包)
    - [g.安装 python](#g安装-python)
    - [h.安装 magma-cuda](#h安装-magma-cuda)
    - [i.配置路径](#i配置路径)
    - [j.安装一些pip 包](#j安装一些pip-包)
    - [k.安装mpi4py （需下载到本地安装，pip 安装可能因为版本兼容问题报错）](#k安装mpi4py-需下载到本地安装pip-安装可能因为版本兼容问题报错)
    - [l.安装pytorch, 版本可替换， 需先下载项目到本地，国内安装容易因为网速原因，造成终止, pytorch git clone 过程中可能有些子包下载过程中会终止。可以多 git clone 几次](#l安装pytorch-版本可替换-需先下载项目到本地国内安装容易因为网速原因造成终止-pytorch-git-clone-过程中可能有些子包下载过程中会终止可以多-git-clone-几次)
    - [m.安装apex](#m安装apex)
    - [n.安装deepspeed](#n安装deepspeed)
    - [o.安装NCCL(可选，pytorch 已自带)](#o安装nccl可选pytorch-已自带)
    - [p.配置网络端口、公钥和ssh](#p配置网络端口公钥和ssh)
  - [6.构建docker 镜像](#6构建docker-镜像)
    - [a.方式一.  pull 镜像](#a方式一--pull-镜像)
    - [b.方式二.  构建镜像](#b方式二--构建镜像)
- [二. 在每个机器节点构建容器](#二-在每个机器节点构建容器)
- [三. 互信机制设置](#三-互信机制设置)
  - [1. 公钥生成默认docker 镜像创建时已生成，如不存在，则在shell 端输入](#1-公钥生成默认docker-镜像创建时已生成如不存在则在shell-端输入)
  - [2.将各节点容器生成的公钥文件](#2将各节点容器生成的公钥文件)
  - [3.免密登陆](#3免密登陆)
  - [4.测试](#4测试)
- [四.  分布式训练测试](#四--分布式训练测试)
  - [a.配置hostfile（hostfile 中的V100-1 与~/.ssh/config 对应）:](#a配置hostfilehostfile-中的v100-1-与sshconfig-对应)
  - [b. 配置glm 文件，各节点配置code 和数据，要求路径相同（也可共同访问云端共享文件）](#b-配置glm-文件各节点配置code-和数据要求路径相同也可共同访问云端共享文件)
  - [c. cmd](#c-cmd)

# 一.  Docker

## 1.安装docker

```shell
#由于Ubuntu里apt官方库里的docker版本可能比较低，因此先用下面的命令行卸载旧版本
apt-get remove docker docker-engine docker-ce docker.io

#更新apt包索引
apt-get update

#执行下列命令行，使apt可以通过HTTPS协议去使用存储库
apt-get install -y apt-transport-https ca-certificates curl software-properties-common

#添加Docker官方提供的GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

#设置stable存储库
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

#再次更新apt包索引
apt-get update

#安装最新版本的docker-ce
apt-get install -y docker-ce
```

## 2.Docker 换源

(https://xxxx.mirror.aliyuncs.com) 为自己的docker源仓库

```shell
mkdir -p /etc/docker
tee /etc/docker/daemon.json 
-'EOF'
{
  "registry-mirrors": ["https://xxxx.mirror.aliyuncs.com"]
}
EOF

systemctl daemon-reload
systemctl restart docker
```

## 3.安装显卡驱动（如已装可跳过）

```shell
#检验是否存在Nvidia驱动
dpkg --list | grep nvidia-*
#或者执行 cat /proc/driver/nvidia/version

#如果不存在Nvidia驱动，则需要安装
#执行ubuntu-drivers devices，查看推荐驱动版本
ubuntu-drivers devices
#如果“Command 'ubuntu-drivers' not found”，执行apt-get install ubuntu-drivers-common

#安装推荐的驱动版本
apt-get install nvidia-driver-版本号

#检验是否安装成功
nvidia-smi
#如果出现"NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."错误
#执行apt install dkms
#查看版本号ls /usr/src | grep nvidia
#dkms install -m nvidia -v + 版本号
#注意：安装完成后，可能需要重新启动服务器

```
## 4.配置nvidia-docker源：

```shell
#添加源
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

#安装nvidia-docker2和依赖，安装的过程中，选择“默认”
apt-get update
apt-get install -y nvidia-docker2
````

修改/etc/docker/daemon.json，添加相关信息

```text
"runtimes": {
   "nvidia": {
       "path": "/usr/bin/nvidia-container-runtime",
       "runtimeArgs": []
   }
}
```
/etc/docker/daemon.json最终内容

```json
{
 "registry-mirrors": ["https://xxxx.mirror.aliyuncs.com"],
 "runtimes": {
     "nvidia": {
         "path": "/usr/bin/nvidia-container-runtime",
         "runtimeArgs": []
     }
  }
}
```

重启docker服务

```shell
systemctl daemon-reload
systemctl restart docker
```

## 5.制作dockerfile

### a.拉取nvidia 基础镜像, 创建临时文件夹（容器内，镜像创建完成后，删除）

```dockerfile
#pull base image
FROM nvidia/cuda:10.2-devel-ubuntu18.04 
#maintainer
MAINTAINER deepspeed <gqwang@baai.ac.cn>

##############################################################################
#Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}
```

### b.配置apt 安装源,并安装一些linux 系统常用基础包

```dockerfile
##############################################################################
#Installation/Basic Utilities
##############################################################################
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN  sed -i s@/security.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev libsndfile-dev \
        libcupti-dev \
        libjpeg-dev \
        libpng-dev \
        screen jq psmisc dnsutils lsof musl-dev systemd
```      
      
### c.  安装最新版git(创建镜像clone 安装包)

```dockerfile
##############################################################################
#Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version
```
### d. 安装  Mellanox OFED, 由于网络问题，推荐安装包下到本地后，再执行dockerfile

```dockerfile
##############################################################################
#install Mellanox OFED
#dwonload from  https://www.mellanox.com/downloads/ofed/MLNX_OFED-5.1-2.5.8.0/MLNX_OFED_LINUX-5.1-2.5.8.0-ubuntu18.04-x86_64.tgz
##############################################################################
RUN apt-get install -y libnuma-dev  libnuma-dev libcap2
ENV MLNX_OFED_VERSION=5.1-2.5.8.0
COPY MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64.tgz ${STAGE_DIR}
RUN cd ${STAGE_DIR} && \
    tar xvfz MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64.tgz && \
    cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64 && \
    PATH=/usr/bin:$PATH ./mlnxofedinstall --user-space-only --without-fw-update --umad-dev-rw --all -q && \
    cd ${STAGE_DIR} && \
    rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64*
```  

### e. 安装 nv_peer_mem

```dockerfile
##############################################################################
#Install nv_peer_mem
##############################################################################
#COPY nv_peer_memory ${STAGE_DIR}/nv_peer_memory (without net)
############try for more times #################

ENV NV_PEER_MEM_VERSION=1.1
ENV NV_PEER_MEM_TAG=1.1-0
RUN git clone https://github.com/Mellanox/nv_peer_memory.git --branch ${NV_PEER_MEM_TAG} ${STAGE_DIR}/nv_peer_memory
RUN cd ${STAGE_DIR}/nv_peer_memory && \
    ./build_module.sh && \
    cd ${STAGE_DIR} && \
    tar xzf ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_VERSION}.orig.tar.gz && \
    cd ${STAGE_DIR}/nvidia-peer-memory-${NV_PEER_MEM_VERSION} && \
    apt-get update && \
    apt-get install -y dkms && \
    dpkg-buildpackage -us -uc && \
    dpkg -i ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_TAG}_all.deb 
```

### f. 安装openmpi, 需先安装libevent 依赖包

```dockerfile
###########################################################################
#Install libevent && OPENMPI
#https://www.open-mpi.org/software/ompi/v4.0/
##############################################################################
ENV OPENMPI_BASEVERSION=4.0
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.5
COPY openmpi-4.0.5.tar.gz  ${STAGE_DIR}
COPY libevent-2.0.22-stable.tar.gz  ${STAGE_DIR}
RUN cd ${STAGE_DIR} && \
    tar zxvf libevent-2.0.22-stable.tar.gz && \
    cd libevent-2.0.22-stable && \
    ./configure --prefix=/usr && \
    make && make install
RUN cd ${STAGE_DIR} && \
    tar --no-same-owner -xzf openmpi-4.0.5.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install  && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    #Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ${STAGE_DIR} && \
    rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
#Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun
```
  
### g.安装 python

```dockerfile
###########################################################################
#Install python
##############################################################################
ARG PYTHON_VERSION=3.8
RUN curl -o ~/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing
```
  
### h.安装 magma-cuda

```dockerfile
###########################################################################
#Install magma-cuda
##############################################################################
COPY magma-cuda102-2.5.2-1.tar.bz2   ${STAGE_DIR} 
RUN  cd ${STAGE_DIR} && \
     /opt/conda/bin/conda install -y -c pytorch --use-local magma-cuda102-2.5.2-1.tar.bz2  && \
     /opt/conda/bin/conda clean -ya
####optional#####
#RUN  /opt/conda/bin/conda install -y -c pytorch  magma-cuda102  && \
#/opt/conda/bin/conda clean -ya
```
  
### i.配置路径

```dockerfile
###########################################################################
#Export path
##############################################################################
ENV PATH /opt/conda/bin:$PATH
RUN echo "export PATH=/opt/conda/bin:\$PATH" >> /root/.bashrc
RUN pip install --upgrade pip setuptools
RUN wget https://tuna.moe/oh-my-tuna/oh-my-tuna.py && python oh-my-tuna.py
```
  
### j.安装一些pip 包

```dockerfile
###########################################################################
#Install some Packages
##############################################################################
RUN pip install psutil \
                yappi \
                cffi \
                ipdb \
                h5py \
                pandas \
                matplotlib \
                py3nvml \
                pyarrow \
                graphviz \
                astor \
                boto3 \
                tqdm \
                sentencepiece \
                msgpack \
                requests \
                pandas \
                sphinx \
                sphinx_rtd_theme \
                sklearn \
                scikit-learn \
                nvidia-ml-py3 \
                nltk \
                rouge \
                filelock \
                fasttext \
                rouge_score \
                cupy-cuda102\
                setuptools==60.0.3
```
  
### k.安装mpi4py （需下载到本地安装，pip 安装可能因为版本兼容问题报错）

```dockerfile
##############################################################################
#Install mpi4py
##############################################################################
RUN apt-get update && \
 apt-get install -y mpich
COPY mpi4py-3.1.3.tar.gz ${STAGE_DIR}
RUN cd ${STAGE_DIR} && tar zxvf mpi4py-3.1.3.tar.gz && \
 cd mpi4py-3.1.3 &&\
 python setup.py build && python setup.py install
```

### l.安装pytorch, 版本可替换， 需先下载项目到本地，国内安装容易因为网速原因，造成终止, pytorch git clone 过程中可能有些子包下载过程中会终止。可以多 git clone 几次

```dockerfile
##############################################################################
#PyTorch
#clone (may be time out because of the network problem)
#RUN git clone --recursive https://github.com/pytorch/pytorch --branch v1.8.1 /opt/pytorch
#RUN cd /opt/pytorch && git checkout -f v1.8.1 && \
#git submodule sync && git submodule update -f --init --recursive
##############################################################################
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"

COPY opt/pytorch /opt/pytorch
ENV NCCL_LIBRARY=/usr/lib/x86_64-linux-gnu
ENV NCCL_INCLUDE_DIR=/usr/include
RUN cd /opt/pytorch && TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" USE_SYSTEM_NCCL=1 \
    pip install -v . && rm -rf /opt/pytorch



##############################################################################
#Install vision
#RUN git clone https://github.com/pytorch/vision.git /opt/vision
##############################################################################
COPY vision /opt/vision
RUN cd /opt/vision  && pip install -v . && rm -rf /opt/vision
ENV TENSORBOARDX_VERSION=1.8
RUN pip install tensorboardX==${TENSORBOARDX_VERSION}
```

### m.安装apex

```dockerfile
###########################################################################
#Install apex
###########################################################################
#RUN git clone https://github.com/NVIDIA/apex ${STAGE_DIR}/apex
COPY apex ${STAGE_DIR}/apex
RUN cd ${STAGE_DIR}/apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
    && rm -rf ${STAGE_DIR}/apex
```

### n.安装deepspeed

```dockerfile
 ############################################################################
#Install deepSpeed
#############################################################################
RUN pip install  py-cpuinfo
RUN apt-get install -y libaio-dev
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
#COPY DeepSpeed ${STAGE_DIR}/DeepSpeed
RUN cd ${STAGE_DIR}/DeepSpeed &&  \
    git checkout . && \
    DS_BUILD_OPS=1 ./install.sh -r
RUN rm -rf ${STAGE_DIR}/DeepSpeed
RUN python -c "import deepspeed; print(deepspeed.__version__)"
```

### o.安装NCCL(可选，pytorch 已自带)

```dockerfile
############################################################################
#Install nccl
#############################################################################

#COPY  nccl-local-repo-ubuntu1804-2.9.6-cuda10.2_1.0-1_amd64.deb ${STAGE_DIR}
#RUN cd ${STAGE_DIR} &&\
#sudo dpkg -i nccl-local-repo-ubuntu1804-2.9.6-cuda10.2_1.0-1_amd64.deb &&\
#sudo apt install -y   libnccl2 libnccl-dev
#RUN apt install -y --allow-downgrades --no-install-recommends --allow-change-held-packages  libnccl2=2.9.6-1+cuda10.2 libnccl-dev=2.9.6-1+cuda10.2
#ENV NCCL_VERSION=2.9.6
```

### p.配置网络端口、公钥和ssh

```dockerfile
#############################################################################
#Set SSH Config
#############################################################################
RUN apt-get install openssh-server

ARG SSH_PORT=6001
#RUN echo 'root:NdjeS+-4gEPmq}D' | chpasswd
#Client Liveness & Uncomment Port 22 for SSH Daemon
RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
RUN mkdir -p /var/run/sshd && cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
    sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
    sed "0,/^Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
EXPOSE ${SSH_PORT}
#Set SSH KEY
RUN printf "#StrictHostKeyChecking no\n#UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
 ssh-keygen -t rsa -f ~/.ssh/id_rsa -N "" && cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
   chmod og-wx ~/.ssh/authorized_keys
CMD service ssh start
```


## 6.构建docker 镜像

### a.方式一.  pull 镜像

```shell
#远程拉取
docker pull deepspeed/cuda102
#本地读取
docker load --input deepspeed-cuda102.tar.gz
```
### b.方式二.  构建镜像

```shell
docker build -f cuda102.dockerfile  -t deepspeed/cuda102:1221 .
#cuda102.dockerfile 参考 dockerfile 文件制作流程
```   
# 二. 在每个机器节点构建容器

```shell
# 创建容器（nvidia-docker），hostname = 容器内部host名称，
# network= host 与数组机共享
# ipc=host, 集群训练时，需按此设置
# shm_size 共享内存，name 容器外部名
# --gpus 指定gpu
# 多数据卷：-v 本地文件夹:容器内文件夹 -v 本地文件夹:容器内文件夹 -v 本地文件夹:容器内文件夹  deepspeed/cuda102:1221 镜像名：tag
nvidia-docker run -id  --hostname=glm_dist16  --network=host --ipc=host --shm-size=16gb --name=glm_dist16   --gpus '"device=0,1,2,3"' -v /data1/docker/containers:/data  deepspeed/cuda102:1221
```

```shell
#拉取镜像
docker pull nvidia/cuda:cuda版本-runtime-ubuntu版本
#拉取镜像举例
docker pull nvidia/cuda:10.1-runtime-ubuntu18.04
#创建容器（普通docker），比如多端口号：-p 22:22 -p 80:80 -p 8080:8080，多数据卷：-v 文件夹:文件夹 -v 文件夹:文件夹 -v 文件夹:文件夹
docker run -id --name=容器名 -p 宿主机端口号:容器内端口号 -e TZ=Asia/Shanghai -v 宿主机文件夹:容器内文件夹 镜像:版本号
#创建容器（普通docker）举例
docker run -id --name=test -p 80:80 -e TZ=Asia/Shanghai -v /data:/data mysql:5.7
#创建容器（nvidia-docker），支持多映射，比如多端口号：-p 22:22 -p 80:80 -p 8080:8080，多数据卷：-v 文件夹:文件夹 -v 文件夹:文件夹 -v 文件夹:文件夹
nvidia-docker run -id --name=容器名 -p 宿主机端口号:容器内端口号 -e TZ=Asia/Shanghai --shm-size=大小 -v 宿主机文件夹:容器内文件夹 镜像:版本号
#创建容器（nvidia-docker）举例
docker run -id --name=test -p 80:80 -e TZ=Asia/Shanghai --shm-size=8gb -v /data:/data nvidia/cuda:10.1-runtime-ubuntu18.04
#进入容器（普通docker）
docker exec -it 容器名 /bin/bash
#进入容器（nvidia-docker）
nvidia-docker exec -it 容器名 /bin/bash
#查看已有镜像
docker images
#删除镜像
docker rmi 镜像名/镜像id
#查看正在运行的容器
docker ps
#查看历史容器，包含正在运行和已经关闭的
docker ps -a
#停止正在运行的容器
docker stop 容器名/容器id
#删除容器，如果待删除的容器正在运行，需要先停止再删除
docker rm 容器名/容器id
```

# 三. 互信机制设置

## 1. 公钥生成默认docker 镜像创建时已生成，如不存在，则在shell 端输入

```shell
 ssh-keygen -t rsa -C "example@com.cn"
```

## 2.将各节点容器生成的公钥文件

~/.ssh/id_rsa.pub
中的内容收集，并同步到各机器的文件
~/.ssh/authorized_keys

## 3.免密登陆

配置各节点容器port : vi /etc/ssh/sshd_config , 将port 注释取消，并设值，统一节点不同容器port 需要不一样
如下配置各节点 host 文件 ：vi ~/.ssh/config 复制各节点host 登陆信息, 并同步到各节点

```text
Host V100-1
    Hostname 172.31.32.29
    Port 6001
    User root
Host V100-2
    Hostname 172.31.32.40
    Port 6002
    User root
```

## 4.测试

```shell
ssh V100-1
```

# 四.  分布式训练测试

## a.配置hostfile（hostfile 中的V100-1 与~/.ssh/config 对应）:

```text
V100-1 slots=4
V100-2 slots=4
V100-3 slots=4
.....
```
## b. 配置glm 文件，各节点配置code 和数据，要求路径相同（也可共同访问云端共享文件）

## c. cmd

```shell
bash config/start_scripts/generate_block.sh  config/config_tasks/model_blocklm_large_chinese.sh
```

