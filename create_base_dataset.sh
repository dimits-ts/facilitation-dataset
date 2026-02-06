mkdir -p "logs"
LOG_FILE="logs/pefk.log"
touch $LOG_FILE

cd scripts/base_dataset
mkdir -p "../datasets"
mkdir -p "../downloads"

# Accept datasets from command line arguments
[[ $# -eq 0 ]] && echo "Usage: $0 dataset1 [dataset2...] [dataset3...]" && exit 1

datasets=("$@")

for datasetname in "${datasets[@]}"; do
    echo "Downloading ${datasetname}..."
    bash "$datasetname.sh"
    echo "Processing ${datasetname}..."
    python -m "scripts.base_dataset.${datasetname}"
    echo "Exported $datasetname.csv"
done

# combine datasets into one_
echo "Processing final dataset..."
python postprocessing.py
echo  "Finished dataset construction."


cd ..
cd ..
# clean-up to release disk space_
echo "Cleaning up downloads directory..."
rm -r "downloads"
echo "Cleaning up intermediate datasets..."
rm -r "datasets"

python scripts/quality_dims/toxicity_annotate.py \
    --input_csv pefk.csv \
    --output_path output_datasets/pefk_toxicity.csv \
    --api_key_file=perspective.key