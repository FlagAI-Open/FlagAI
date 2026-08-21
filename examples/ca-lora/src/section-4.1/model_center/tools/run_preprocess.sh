for ((i=$1; i<$2; i++)); do
{
    python3 /mnt/sfs_turbo/ModelCenter/src/tools/preprocess_cpm1_lm.py --uid $i
}
done

