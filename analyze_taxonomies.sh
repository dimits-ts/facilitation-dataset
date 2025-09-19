#!/bin/bash
LOG_FILE="logs/pefk_taxonomies.log"

python scripts/annotate_taxonomies.py --mod_probability_file=output_datasets/pefk_mod.csv --dataset_file=pefk.csv --taxonomy_file=taxonomies/config/ta
xonomy.yaml --prompt_file=taxonomies/config/prompt.txt --output_dir=taxonomies/output --m
od_probability_thres=0.6 --logs_dir=logs/taxonomies_llm | tee "../$LOG_FILE"

python scripts/taxonomy_train.py --dataset_path pefk.csv --output_dir checkpoints/taxonomies --logs_dir=logs/taxonomies_training --labels_dir=taxonomies/output