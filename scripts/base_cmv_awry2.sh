#!/bin/bash
DOWNLOAD_DIR="../downloads/cmv_awry2"
MASTER_ZIP_PATH="$DOWNLOAD_DIR/cmv_awry2.zip"
URL="https://zissou.infosci.cornell.edu/convokit/datasets/conversations-gone-awry-cmv-corpus/conversations-gone-awry-cmv-corpus.zip"

mkdir -p "$DOWNLOAD_DIR"
wget --no-verbose -nc -O "$MASTER_ZIP_PATH" "$URL"
unzip -u "$MASTER_ZIP_PATH" -d "$DOWNLOAD_DIR"
rm "$MASTER_ZIP_PATH"