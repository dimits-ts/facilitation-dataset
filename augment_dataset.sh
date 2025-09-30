#!/bin/bash
LOG_FILE="logs/pefk_augmented.log"

echo "Training moderation detection model..."
python scripts/facilitation_train.py \
    --output_dir=checkpoints/mod/all \
    --logs_dir=logs/mod/all \
    --dataset_path=pefk.csv \
    --datasets=ceri,fora,wikitactics,whow,umod,iq2 | tee "../$LOG_FILE"

echo "Detecting moderator comments in dataset..."
python scripts/facilitation_inference.py \
        --model_dir checkpoints/mod/all/best_model \
        --source_dataset_path=pefk.csv \
        --destination_dataset_path=output_datasets/pefk_mod.csv | tee "../$LOG_FILE"

echo "Starting analysis..."
python scripts/facilitation_analysis.py \
        --dataset_file pefk.csv \
        --mod_probability_file=output_datasets/pefk_mod.csv \
        --mod_probability_thres=0.6

echo "Finished facilitation analysis."