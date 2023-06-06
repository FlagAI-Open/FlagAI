for a in `cat $1 | awk -F " " '{print $1}'`; do pdsh -f 1024 -w ssh:$a "pip uninstall bmtrain -y"; done
for a in `cat $1 | awk -F " " '{print $1}'`; do pdsh -f 1024 -w ssh:$a "pip install bmtrain"; done
