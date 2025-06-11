#!/bin/bash
process_file() {
  file="$1"
  filename=$(basename "$file")
  stem="${filename%.gz}"
  output_file="$DOWNLOAD_DIR/${stem}.json"

  if [ ! -f "$output_file" ]; then
      echo "Unzipping $file -> $output_file"
      if gunzip -c "$file" > "$output_file"; then
          rm -f "$file"
      else
          echo "Failed to unzip $file" >&2
          rm -f "$output_file" 
      fi
  else
      echo "File $output_file already exists, skipping..."
  fi
}


DOWNLOAD_DIR="downloads/cmv_awry2"
MASTER_ZIP_PATH="$DOWNLOAD_DIR/cmv_awry2.zip"
URL="https://figshare.com/ndownloader/articles/21077989/versions/1"
COMBINED_JSON="$DOWNLOAD_DIR/combined.json"
export -f process_file
export DOWNLOAD_DIR 

if [ -f "$COMBINED_JSON" ]; then
    echo "Combined JSON file already exists. Exiting."
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

echo "Extracting inner archives..."
# Parallel gunzip extraction
find "$DOWNLOAD_DIR" -type f -name "*.gz" | parallel -j5 process_file

# Tange, O. (2025, April 22). GNU Parallel 20250422 ('Tariffs').
# Zenodo. https://doi.org/10.5281/zenodo.15265748

echo "Combining JSON files..."

COMBINED_JSON="$DOWNLOAD_DIR/combined.json"
echo "[" > "$COMBINED_JSON"

first=1
find "$DOWNLOAD_DIR" -maxdepth 1 -type f -iname '*.json' ! -name 'combined.json' -print0 | while IFS= read -r -d '' file; do
    if [ $first -eq 1 ]; then
        first=0
    else
        echo "," >> "$COMBINED_JSON"
    fi
    cat "$file" >> "$COMBINED_JSON"
done

echo "]" >> "$COMBINED_JSON"

# Optionally delete the originals
find "$DOWNLOAD_DIR" -maxdepth 1 -type f -iname '*.json' ! -name 'combined.json' -delete

echo "Finished"