#!/bin/bash
LOG_FILE="logs/pefk_taxonomies.log"

python scripts/annotate_taxonomies.py --mod_probability_file=output_datasets/pefk_mod.csv --dataset_file=pefk.csv --taxonomy_file=taxonomies/config/ta
xonomy.yaml --prompt_file=taxonomies/config/prompt.txt --output_dir=taxonomies/output --m
od_probability_thres=0.6 --logs_dir=logs | tee "../$LOG_FILE"