#ckpt=$(awk 'NR==1 {print $1}' /share/project/64node-bmt-flashatten/checkpoints/Aquila-30b-64n8g-reinit/latest_iteration.txt)
for ckpt in {61000,62000,63000,64000};do
	sshpass -p "Dg%xoL%tDaCJSg4I" scp -r ubuntu@36.103.236.162:/data2/checkpoints/Aquila-7b-24n8g-V3-reload52/$ckpt ./
	python generate_valid_loss.py $ckpt
	python generate_valid_loss_all.py $ckpt
	rm -rf $ckpt
done
