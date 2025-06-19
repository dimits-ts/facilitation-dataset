#!/bin/bash
PEFK_PATH="../pefk.csv"
LOG_FILE="../pefk.log"

if [ -f "$PEFK_PATH" ]; then 
    echo "Combined dataset already exists. Exiting..."
    exit 0
fi

# enable logging
exec 1 | tee ${LOG_FILE}
exec 2 | tee ${LOG_FILE}

# download datasets to ../downloads, then export them to ../datasets
declare -a datasetnames=("ceri" "cmv_awry2" "umod" "vmd" "wikiconv" "wikidisputes" "wikitactics")

for datasetname in "${datasetnames[@]}"; do
    echo "Downloading ${datasetname}.sh"
    bash "$datasetname.sh"
    echo "Processing ${datasetname}.py"
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