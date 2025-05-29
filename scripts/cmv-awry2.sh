#!/bin/bash
DOWNLOAD_DIR="downloads/cmv_awry2"
DOWNLOAD_PATH="$DOWNLOAD_DIR/cmv_awry2.zip"

URL="https://figshare.com/ndownloader/articles/21077989/versions/1"
  
# what the hell is this dataset
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
   
mkdir -p $(dirname $DOWNLOAD_PATH)
  
if [ ! -f $DOWNLOAD_PATH ]; then
  echo "Downloading..."
  wget -O $DOWNLOAD_PATH $URL
else
  echo "Download already exists. Skipping download."
fi
  
echo "Extracting..."
unzip -u $DOWNLOAD_PATH -d $DOWNLOAD_DIR
for file in `find . -type f -name "*.gz"`; do
    echo "Unzipping $file..."
    STEM=$(basename "${file}" .gz)
    OUTPUT_FILE="$DOWNLOAD_DIR/$STEM.json"
    touch $OUTPUT_FILE
    gunzip -c "${file}" > $OUTPUT_FILE
done
echo "Finished"