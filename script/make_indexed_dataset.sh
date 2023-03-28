export PYTHONPATH=YOUR_FLAGAI_HOME

PREPROCESS_DATA_TOOL=$PYTHONPATH/flagai/data/dataset/indexed_dataset/preprocess_data_args.py
INPUT_DIR=YOUR_INPUT_DIR
OUTPUT_DIR=YOUR_OUTPUT_DIR

TOKENIZER_DIR=YOUR_TOKENIZER_DIR
TOKENIZER_NAME=YOUR_TOKENIZER_NAME

file=YOUR_INPUT_FILE
output_prefix=YOUR_OUTPUT_PREFIX
python $PREPROCESS_DATA_TOOL --input $file --output-prefix $output_prefix --workers 16 --chunk-size 256 --model-name $TOKENIZER_NAME --model-dir $TOKENIZER_DIR
