#ckpt=$(awk 'NR==1 {print $1}' /share/project/64node-bmt-flashatten/checkpoints/Aquila-30b-64n8g-reinit/latest_iteration.txt)
for ckpt in {17000,22000,27000,32000,37000,42000,47000,52000,57000,62000};do
	sshpass -p "Dg%xoL%tDaCJSg4I" scp -r ubuntu@36.103.236.162:/data2/checkpoints/Aquila-7b-24n8g-V3/$ckpt ./
	python generate_valid_loss_old.py $ckpt
	python generate_valid_loss_all_old.py $ckpt
	rm -rf $ckpt
done
