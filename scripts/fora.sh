#!/bin/bash

INPUT_FILE="../downloads_external/fora.zip"
OUTPUT_DIR="../downloads/fora"

if [ ! -f $INPUT_FILE ]; then
    echo "The Fora corpus does not seem to be available locally at $(readlink -f "$INPUT_FILE")"
    echo "Since the corpus is closed-access, you need to ask for permission to download it."
    echo "If you have it available, place the zip in the path printed above and rerun the script"
    echo "More information here: https://github.com/schropes/fora-corpus"
else
    unzip -u $INPUT_FILE -d $OUTPUT_DIR
fi