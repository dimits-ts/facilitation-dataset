#!/bin/bash
URL="https://datasets-server.huggingface.co/rows?dataset=gkaradzhov%2FDeliData&config=default&split=train&offset=0"
DOWNLOAD_DIR="downloads/delidata"

mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR && { curl -o "delidata.json" -X GET $URL ; cd -; }
#mv "$DOWNLOAD_DIR/rows" "$DOWNLOAD_DIR/delidata.json"