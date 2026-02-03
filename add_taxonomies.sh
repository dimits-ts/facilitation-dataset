#!/bin/bash
python scripts/taxonomy/llm_annotate.py \
    --mod_probability_file=output_datasets/pefk_mod.csv \
    --dataset_file=pefk.csv --taxonomy_file=taxonomies/config/taxonomy.yaml \
    --prompt_file=taxonomies/config/prompt.json \
    --output_dir=taxonomies/output \
    --mod_probability_thres=0.6 \
    --logs_dir=logs/taxonomies_llm

python scripts/taxonomy/trans_train.py \
    --dataset_path pefk.csv \
    --output_dir checkpoints/taxonomies/fora \
    --logs_dir=logs/taxonomies_training/fora \
    --sub_dataset_name=fora

python scripts/taxonomy/trans_train.py \
    --dataset_path pefk.csv \
    --output_dir checkpoints/taxonomies/whow \
    --logs_dir=logs/taxonomies_training/whow \
    --sub_dataset_name=whow

python scripts/taxonomy/trans_inference.py \
    --model_dir=checkpoints/taxonomies \
    --dataset_path=pefk.csv \
    --labels_dir=taxonomies/output \
    --output_csv=output_datasets/taxonomies.csv \
    --mod_probability_path=output_datasets/pefk_mod.csv

python scripts/taxonomy/analysis.py \
    --res_csv_path=logs/taxonomies_training/res.csv \
    --graphs_dir=graphs \
    --dataset_path=pefk.csv \
    --label_dir=taxonomies/output