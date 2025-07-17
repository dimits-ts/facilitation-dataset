#!/bin/bash
PEFK_PATH="../pefk.csv"
MOD_MODEL_PATH="../checkpoints/mod/all/best_model"

if [ -f "$PEFK_PATH" ]; then
    echo "Combined dataset already exists. Exiting..."
    exit 0
fi

mkdir -p "../datasets"
mkdir -p "../downloads"

# Accept datasets from command line arguments
[[ $# -eq 0 ]] && echo "Usage: $0 dataset1 [dataset2...] [dataset3...]" && exit 1

datasets=("$@")

for datasetname in "${datasets[@]}"; do
    echo "Downloading ${datasetname}..."
    bash "$datasetname.sh"
    echo "Processing ${datasetname}..."
    python "$datasetname.py"
    echo "Exported $datasetname.csv"
done

# combine datasets into one_
echo "Processing final dataset..."
python postprocessing.py

echo  "Finished dataset construction."
python dataset_analysis.py

# clean-up to release disk space_
echo "Cleaning up downloads directory..."
#rm -r "../downloads"
echo "Cleaning up intermediate datasets..."
#rm -r "../datasets"

if [ ! -d "$MOD_MODEL_PATH" ]; then
    echo "Moderator trained model directory does not exist. Training model..."
    python ../scripts/model_train.py \
        --output_dir=checkpoints/mod/all \
        --logs_dir=logs/mod/all \
        --dataset_path=pefk.csv \
        --datasets=ceri,fora,wikitactics,whow,umod,iq2
else
    echo "Moderator trained model directory already exists. Skipping training..."
fi

echo "Detecting moderator comments in dataset..."
python scripts/model_inference.py \
        --model_dir checkpoints/mod/all/best_model \
        --source_dataset_path=pefk.csv \
        --destination_dataset_path=pefk_mod.csv

echo "Finished."
