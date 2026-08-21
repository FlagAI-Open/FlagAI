set -ex

# Number of training instances per label.
K=16

# Data directory for k-shot splits.
DATA_DIR="data/k-shot"

# Output directory where results will be written.
OUTPUT_DIR="my_auto_label_mapping"

# Pre-trained model name (roberta-*, bert-*), see Transformers.
MODEL_NAME="roberta-large"

# For auto T + L, we first generate automatic templates. Then, for each template, we
# generate automatic labels. Finally we will train all auto template X auto labels and
# select the best (based on dev). If we are doing this, then we must specify the auto T
# results, and load the top n per result.
LOAD_TEMPLATES="false"
TEMPLATE_DIR="auto_template/"
NUM_TEMPLATES=10

# Filter options to top K words (conditional) per class.
K_LIKELY=100

# Special case: we may need to further re-rank based on K-NN.
K_NEIGHBORS=30

# How many label mappings per template to keep at the end.
N_PAIRS=100

TASKS="SST-2"

SEEDS="100 13 21 42 87"

TASK_EXTRA=""

for TASK in $TASKS; do
    for SEED in $SEEDS; do
        case $TASK in
            CoLA)
                TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
                MAPPING="{'0':'incorrect','1':'correct'}"
                ;;
            SST-2)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{'0':'terrible','1':'great'}"
                ;;
            MRPC)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            QQP)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            STS-B)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            MNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
                ;;
            SNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
                ;;
            QNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'not_entailment':'No','entailment':'Yes'}"
                ;;
            RTE)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'not_entailment':'No','entailment':'Yes'}"
                TASK_EXTRA="--first_sent_limit 240"
                ;;
            mr)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            sst-5)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20"
                ;;
            subj)
                TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
                MAPPING="{0:'subjective',1:'objective'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            trec)
                TEMPLATE="*cls**mask*:*+sent_0**sep+*"
                MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
                K_LIKELY=1000
                K_NEIGHBORS=20
                TASK_EXTRA="--first_sent_limit 110 --use_seed_labels"
                ;;
            cr)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            mpqa)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110"
                ;;
        esac

        if [[ $LOAD_TEMPLATES = "true" ]]; then
            FILENAME=$TEMPLATE_DIR/${TASK}/$K-${SEED}.sort.txt
            for TEMPLATE in $(head -n $NUM_TEMPLATES $FILENAME); do
                python tools/generate_labels.py \
                       --overwrite_output_dir \
                       --output_dir /tmp/output \
                       --model_name_or_path $MODEL_NAME \
                       --output_file $OUTPUT_DIR/auto_template/$TASK/$K-$SEED.txt \
                       --append_output_file \
                       --write_template \
                       --template $TEMPLATE \
                       --mapping $MAPPING \
                       --task_name $TASK \
                       --data_dir $DATA_DIR/$TASK/$K-$SEED \
                       --k_likely $K_LIKELY \
                       --k_neighbors $K_NEIGHBORS \
                       --n_pairs $(($N_PAIRS / $NUM_TEMPLATES)) \
                       --max_seq_len 256 \
                       --per_device_eval_batch_size 16 \
                       $TASK_EXTRA
            done
        else
            python tools/generate_labels.py \
                   --overwrite_output_dir \
                   --output_dir /tmp/output \
                   --model_name_or_path $MODEL_NAME \
                   --output_file $OUTPUT_DIR//manual_template/$TASK/$K-$SEED.txt \
                   --template $TEMPLATE \
                   --mapping $MAPPING \
                   --task_name $TASK \
                   --data_dir $DATA_DIR/$TASK/$K-$SEED \
                   --k_likely $K_LIKELY \
                   --k_neighbors $K_NEIGHBORS \
                   --n_pairs $N_PAIRS \
                   --max_seq_len 256 \
                   --per_device_eval_batch_size 16 \
                   $TASK_EXTRA
        fi
    done
done
