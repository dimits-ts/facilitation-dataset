#!/bin/bash
DOWNLOAD_DIR="../../downloads/iq2"
MASTER_ZIP_PATH="$DOWNLOAD_DIR/iq2.zip"
URL="https://tisjune.github.io/datasets/iq2_data_release.zip"

mkdir -p $DOWNLOAD_DIR
wget --no-verbose -nc -O $MASTER_ZIP_PATH $URL
unzip -u "$MASTER_ZIP_PATH" -d "$DOWNLOAD_DIR"