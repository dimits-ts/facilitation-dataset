#!/bin/bash

DOWNLOAD_DIR="../../downloads/wikiconv"
ZIP_ROOT="$DOWNLOAD_DIR/datasets/wikiconv-corpus/corpus-zipped"
URL="https://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/"

mkdir -p "$DOWNLOAD_DIR"
wget --no-verbose -nc -r -nH --no-parent --cut-dirs=1 -P "$DOWNLOAD_DIR" "$URL" -A .zip

echo "Unzipping datasets..."
find "$ZIP_ROOT" -type f -name "full.corpus.zip" | \
  xargs -P 4 -I{} unzip -q -d "$DOWNLOAD_DIR" "{}"
echo "Done."
