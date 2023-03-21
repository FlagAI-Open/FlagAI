mkdir -p current_training
cp /share/project/liuguang/megatron_helper/* /opt/conda/lib/python3.8/site-packages/megatron/data/
ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2 " slots=8"}'|tr -d "addr:" >> current_training/hostfile_llama
