#!/bin/bash
DOWNLOAD_DIR="downloads/cmv_awry2"
MASTER_ZIP_PATH="$DOWNLOAD_DIR/cmv_awry2.zip"
URL="https://figshare.com/ndownloader/articles/21077989/versions/1"


json_file_count=$(find "$DOWNLOAD_DIR" -maxdepth 1 -type f -iname '*.json' | wc -l)
if [ $json_file_count -ge 5 ]; then
    echo "All .json files already there. Exiting."
    exit 0
fi

  
# what the hell is this dataset
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
   
mkdir -p $(dirname $MASTER_ZIP_PATH)
  
echo "Downloading..."
wget -nc -O $MASTER_ZIP_PATH $URL

echo "Extracting..."
unzip -u $MASTER_ZIP_PATH -d $DOWNLOAD_DIR
rm $MASTER_ZIP_PATH # remove large .zip file

for file in `find . -type f -name "*.gz"`; do
  STEM=$(basename "${file}" .gz)
  OUTPUT_FILE="$DOWNLOAD_DIR/$STEM.json"
  if [! -f $OUTPUT_FILE]; then
    echo "Unzipping $file..."
    touch $OUTPUT_FILE
    gunzip -c $file > $OUTPUT_FILE
    rm $file # remove intermediate .gz archive
  else
    echo "File $OUTPUT_FILE already exits, skipping..."
  fi
done

echo "Finished"