#!/bin/bash
DOWNLOAD_DIR="downloads/vmd"
URL="https://github.com/dimits-ts/synthetic_moderation_experiments/raw/refs/heads/master/data/datasets/main/main.zip"

mkdir -p $DOWNLOAD_DIR
wget -nc -O "$DOWNLOAD_DIR/vmd.zip" $URL
unzip "$DOWNLOAD_DIR/vmd.zip" -d $DOWNLOAD_DIR
rm "$DOWNLOAD_DIR/vmd.zip" 