#!/bin/bash

DATA_PATH="./datasets/cmv_awry2/"
INTERMEDIATE_DIR="./datasets/cmv_awry_tmp"
DOWNLOAD_DIR="./downloads"

DOWNLOAD_PATH="$DOWNLOAD_DIR/cmv_awry2.zip"
URL="https://figshare.com/ndownloader/articles/21077989/versions/1"

# what the hell is this dataset
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

mkdir -p $DATA_PATH $DOWNLOAD_PATH
if [ ! -f "$DOWNLOAD_PATH" ]; then
  echo "Downloading..."
  wget -O "$DOWNLOAD_PATH" "$URL"
else
  echo "Download already exists. Skipping download."
fi

echo "Extracting..."
# intermediate .gz files go here
unzip -u $DOWNLOAD_PATH -d $INTERMEDIATE_DIR

for file in "$INTERMEDIATE_DIR"/*; do 
    echo "Unzipping $file..."
    STEM=$(basename "${file}.gz")
    gunzip -c "${file}" > "$DATA_PATH/${STEM}.json" 
done

# cleanup
#rm -r $INTERMEDIATE_DIR

echo "Finished"