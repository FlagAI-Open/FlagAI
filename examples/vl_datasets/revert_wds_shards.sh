wds_path=LLaVA-SFT-Variations/llava-ov-{0..66}.tar
output_path=LLaVA-SFT-Variations

python revert_wds_shards.py --wds-path $wds_path --output-path $output_path
