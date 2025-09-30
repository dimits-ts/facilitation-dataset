#!/bin/bash
DOWNLOAD_DIR="../downloads/wikidisputes"
DOWNLOAD_PATH="$DOWNLOAD_DIR/wikidisputes.tar.gz"
URL="https://github.com/christinedekock11/wikidisputes/raw/refs/heads/main/data.tar.gz"

mkdir -p $DOWNLOAD_DIR
wget --no-verbose -nc -O $DOWNLOAD_PATH $URL
tar -xvzf $DOWNLOAD_PATH -C $DOWNLOAD_DIR