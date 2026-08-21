script_path=$(pwd)

export PYTHONPATH=/home/ldwang/WorkSpace/FlagAI.ftgreat

PREPROCESS_DATA_TOOL=$PYTHONPATH/script/preprocess_data_flagai_args.py
INPUT_DIR=/share/project/bowen/pretrain_dataset_batch3
OUTPUT_DIR=/share/project/ldwang/data/indexed_dataset/batch1_tok100k

TOKENIZER_DIR=/home/ldwang/WorkSpace/FlagAI.ftgreat/examples/gpt3_pretrain
TOKENIZER_NAME=gpt2_new_100k

cd $OUTPUT_DIR

##
degree=2

i=0
for file in $(ls $INPUT_DIR/cn/wudao_base.jsonl)
do
	echo $file
	part=$(awk 'BEGIN{len=split("'${file}'", vec, "/"); subdir=vec[len-1]; split(vec[len], tuple, "."); print subdir"_"tuple[1];}')
	if [ -f $OUTPUT_DIR/${part}_text_document.idx ];then
		echo "PassBy", $OUTPUT_DIR/${part}_text_document.idx
		continue
	fi
	echo "Processing", $part
	python $PREPROCESS_DATA_TOOL --input $file --output-prefix $part --workers 16 --chunk-size 256 --model-name $TOKENIZER_NAME --model-dir $TOKENIZER_DIR &
	i=`expr $i + 1`
	echo $i
	[ `expr $i % $degree` -eq 0 ] && wait
done

