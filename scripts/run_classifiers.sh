if [ ! -d "$MOD_MODEL_PATH" ]; then
    echo "Moderator trained model directory does not exist. Training model..."
    python ../scripts/facilitation_train.py \
        --output_dir=checkpoints/mod/all \
        --logs_dir=logs/mod/all \
        --dataset_path=pefk.csv \
        --datasets=ceri,fora,wikitactics,whow,umod,iq2
else
    echo "Moderator trained model directory already exists. Skipping training..."
fi

echo "Detecting moderator comments in dataset..."
python scripts/facilitation_inference.py \
        --model_dir checkpoints/mod/all/best_model \
        --source_dataset_path=pefk.csv \
        --destination_dataset_path=output_datasets/pefk_mod.csv

echo "Finished."
