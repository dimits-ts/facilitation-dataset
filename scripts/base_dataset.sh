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
    bash "base_$datasetname.sh"
    echo "Processing ${datasetname}..."
    python "base_$datasetname.py"
    echo "Exported $datasetname.csv"
done

# combine datasets into one_
echo "Processing final dataset..."
python base_postprocessing.py

echo  "Finished dataset construction."
python base_dataset_analysis.py

# clean-up to release disk space_
echo "Cleaning up downloads directory..."
rm -r "../downloads"
echo "Cleaning up intermediate datasets..."
rm -r "../datasets"