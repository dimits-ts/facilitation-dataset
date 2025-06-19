#!/bin/bash
DOWNLOAD_DIR="../downloads/umod"
URL="https://raw.githubusercontent.com/Blubberli/userMod/refs/heads/main/dataset/UMOD_aggregated.csv"

mkdir -p $DOWNLOAD_DIR
wget -nc -O $DOWNLOAD_DIR/"umod.csv" $URL