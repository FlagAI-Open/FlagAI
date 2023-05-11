for a in `cat $1 | awk -F " " '{print $1}'`; do pdsh -f 1024 -w ssh:$a "echo \"root               hard    nofile            1024000\" >> /etc/security/limits.conf"; done
for a in `cat $1 | awk -F " " '{print $1}'`; do pdsh -f 1024 -w ssh:$a "echo \"root               soft    nofile            1024000\" >> /etc/security/limits.conf"; done
