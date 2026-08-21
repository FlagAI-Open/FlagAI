mkdir -p /cdgm0705/hyx/calora-master/checkpoints/logs/t5_squad
mkdir -p /cdgm0705/hyx/calora-master/checkpoints/logs/t5_glue
mkdir -p /cdgm0705/hyx/calora-master/checkpoints/logs/t5_superglue
mkdir -p /cdgm0705/hyx/calora-master/checkpoints/logs/t5_cnndm
pip install -r requirements.txt
pip install scikit-learn
pip install bmtrain==0.2.2
pip install sentencepiece
pip install transformers==4.28.0
pip install cpm-kernels==1.0.11
cd opendelta
pip install -e .
cd ../
pip install rouge
pip install .
export PYTHONPATH="/cdgm0705/hyx/calora-master"
echo $PYTHONPATH
pip list
ls
pwd