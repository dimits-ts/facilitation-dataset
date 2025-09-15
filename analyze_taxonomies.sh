#!/bin/bash
bash create_augmented_dataset.sh

LOG_FILE="logs/pefk_taxonomies.log"

python scripts/annotate_taxonomies.py --mod_probability_file=output_datasets/pefk_mod.csv --dataset_file=pefk.csv --taxonomy_file=llm_classification/ta
xonomy.yaml --prompt_file=llm_classification/prompt.txt --output_dir=taxonomies --m
od_probability_thres=0.6 --logs_dir=logs | tee "../$LOG_FILE"