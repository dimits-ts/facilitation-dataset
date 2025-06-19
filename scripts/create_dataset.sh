#!/bin/bash
PEFK_PATH="../pefk.csv"
LOG_FILE="../pefk.log"

if [ -f "$PEFK_PATH" ]; then 
    echo "Combined dataset already exists. Exiting..."
    exit 0
fi

touch "$LOG_FILE"
# setup logging with timestamps
exec > >(awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }' | tee "$LOG_FILE") 2>&1

mkdir -p "../datasets"
mkdir -p "../downloads"

# download datasets to ../downloads, then export them to ../datasets
declare -a datasetnames=("ceri" "cmv_awry2" "umod" "vmd" "wikiconv" "wikidisputes" "wikitactics")

for datasetname in "${datasetnames[@]}"; do
    echo "Downloading ${datasetname}..."
    bash "$datasetname.sh"
    echo "Processing ${datasetname}..."
    python "$datasetname.py"
done

# combine datasets into one
echo "Combining datasets..."
cd "../datasets"
head -n 1 "$(ls *.csv | head -n 1)" > "$PEFK_PATH" && tail -n +2 -q *.csv >> "$PEFK_PATH"
cd ".."

echo  "Finished dataset construction."

# clean-up to release disk space
echo "Cleaning up downloads directory..."
rm -r "./downloads"
echo "Cleaning up intermediate datasets..."
rm -r "./datasets"

echo "Finished."