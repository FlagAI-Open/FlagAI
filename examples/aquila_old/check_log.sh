cat aquila30b64n8g/202304200833/Aquila-30b-64n8g.yaml.log.txt  |grep Rank\ 0\]\ \ iteration > rank0.log
cat rank0.log |awk -F ' ' '{print $23}' > loss
tail -n10 rank0.log

