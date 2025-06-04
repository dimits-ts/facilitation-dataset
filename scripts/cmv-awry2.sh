#!/bin/bash

DOWNLOAD_DIR="downloads/cmv_awry2"
MASTER_ZIP_PATH="$DOWNLOAD_DIR/cmv_awry2.zip"
URL="https://figshare.com/ndownloader/articles/21077989/versions/1"

# Check if .json files already exist
json_file_count=$(find "$DOWNLOAD_DIR" -maxdepth 1 -type f -iname '*.json' | wc -l)
if [ $json_file_count -ge 5 ]; then
    echo "All .json files already there. Exiting."
    exit 0
fi

# Disable unzip zipbomb protection
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

mkdir -p "$(dirname "$MASTER_ZIP_PATH")"

echo "Downloading..."
wget -nc -O "$MASTER_ZIP_PATH" "$URL"

echo "Extracting master zip..."
unzip -u "$MASTER_ZIP_PATH" -d "$DOWNLOAD_DIR"
rm "$MASTER_ZIP_PATH"

# Parallel gunzip extraction using xargs
find "$DOWNLOAD_DIR" -type f -name "*.gz" | \
  xargs -P 4 -I{} bash -c '
    file="{}"
    stem=$(basename "$file" .gz)
    output_file="'$DOWNLOAD_DIR'/$stem.json"
    if [ ! -f "$output_file" ]; then
        echo "Unzipping $file..."
        gunzip -c "$file" > "$output_file"
        rm "$file"
    else
        echo "File $output_file already exists, skipping..."
    fi
'

echo "Finished"