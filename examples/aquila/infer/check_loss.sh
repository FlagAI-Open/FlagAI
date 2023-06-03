#ckpt=$(awk 'NR==1 {print $1}' /share/project/64node-bmt-flashatten/checkpoints/Aquila-30b-64n8g-reinit/latest_iteration.txt)
sshpass -p "Dg%xoL%tDaCJSg4I" scp -r ubuntu@36.103.236.162:/data2/checkpoints/Aquila-7b-24n8g-V3-reload80/latest_iteration.txt ./
ckpt=$(awk 'NR==1 {print $1}' latest_iteration.txt)
sshpass -p "Dg%xoL%tDaCJSg4I" scp -r ubuntu@36.103.236.162:/data2/checkpoints/Aquila-7b-24n8g-V3-reload80/$ckpt ./
python generate_valid_loss.py $ckpt
python generate_valid_loss_all.py $ckpt
python generate_valid_loss_code.py $ckpt
rm -rf $ckpt
