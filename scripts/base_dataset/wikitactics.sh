#!/bin/bash
DOWNLOAD_DIR="../../downloads/wikitactics"
URL="https://raw.githubusercontent.com/christinedekock11/wikitactics/refs/heads/main/wikitactics.json"

mkdir -p $DOWNLOAD_DIR
wget --no-verbose -nc -O $DOWNLOAD_DIR/"wikitactics.json" $URL