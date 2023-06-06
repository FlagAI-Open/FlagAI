python killall_nodes.py hostfile.Aquila-30b-64n8g
sleep 120
./dist_trigger_docker-flash.sh hostfile.Aquila-30b-64n8g Aquila-30b-64n8g.yaml llama-30b-en Aquila-30b-64n8g-from-scratch
sleep 10
bash prepare_run.sh
tail -f log
