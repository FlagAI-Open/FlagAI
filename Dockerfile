#Change to your base image, such as pytorch1.11+py38
#https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-02.html#rel_21-02
FROM nvcr.io/nvidia/pytorch:21.06-py3
#You can set available pypi sources
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"

ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}
#Ubuntu
RUN apt-get update && apt-get install -y openssh-server && apt-get install -y git
ARG SSH_PORT=6001
#Client Liveness & Uncomment Port 22 for SSH Daemon
RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
RUN mkdir -p /var/run/sshd && cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
    sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
    sed "0,/^Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
EXPOSE ${SSH_PORT}

#Set SSH KEY
RUN mkdir /root/.ssh
RUN printf "#StrictHostKeyChecking no\n#UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
 ssh-keygen -t rsa -f /root/.ssh/id_rsa -N "" && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
   chmod og-wx /root/.ssh/authorized_keys

RUN echo $'Host 127.0.0.1 \n\
    Hostname 127.0.0.1 \n\
    Port 6001 \n\
    StrictHostKeyChecking no \n\
    User root' > /root/.ssh/config
RUN echo $'Host localhost \n\
    Hostname localhost \n\
    Port 6001 \n\
    StrictHostKeyChecking no \n\
    User root' >> /root/.ssh/config

RUN echo "service ssh start" >> /root/.bashrc

#Main deps
RUN pip install tensorboard
RUN pip install sentencepiece
RUN pip install boto3
RUN pip install jieba
RUN pip install ftfy
RUN pip install deepspeed==0.7.7
RUN pip install bmtrain

RUN pip install flagai
#For development usage, you can change as follows
#RUN git clone https://github.com/FlagAI-Open/FlagAI.git && cd FlagAI && python setup.py install

CMD service ssh start && tail -f /dev/null
