#!/bin/bash

DOWNLOAD_DIR="../downloads/wikiconv"
ZIP_ROOT="$DOWNLOAD_DIR/datasets/wikiconv-corpus/corpus-zipped"
URL="https://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/"

# Check if 18 directories exist and match wikiconv-2001 to wikiconv-2018
EXPECTED_DIRS=18
actual_count=$(find "$DOWNLOAD_DIR" -maxdepth 1 -type d -regextype posix-extended -regex ".*/wikiconv-20(0[1-9]|1[0-8])" | wc -l)
if [ "$actual_count" -eq "$EXPECTED_DIRS" ]; then
    echo "All 18 wikiconv directories already exist. Skipping download and unzip."
    exit 0
fi

mkdir -p "$DOWNLOAD_DIR"
wget -nc -r -nH --no-parent --cut-dirs=1 -P "$DOWNLOAD_DIR" "$URL" -A .zip

# Parallel unzipping using xargs with 4 parallel processes (you can adjust -P)
find "$ZIP_ROOT" -type f -name "full.corpus.zip" | \
  xargs -P 4 -I{} unzip {} -q -d "$DOWNLOAD_DIR"

mv "$DOWNLOAD_DIR/datasets/wikiconv-corpus/blocks.json" "$DOWNLOAD_DIR"

# Cleanup
rm -r "$DOWNLOAD_DIR/datasets"
