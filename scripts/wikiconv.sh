DOWNLOAD_DIR="downloads/wikiconv"
ZIP_ROOT="$DOWNLOAD_DIR/datasets/wikiconv-corpus/corpus-zipped"
URL="https://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/"

mkdir -p $DOWNLOAD_DIR
wget -nc -r -nH --no-parent --cut-dirs=1 -P $DOWNLOAD_DIR $URL -A .zip

# for each directory in the directory ./datasets/wikiconv-corpus/corpus-zipped
#   unzip full.corpus.zip
#   move unzipped contents to ./wikiconv

find "$ZIP_ROOT" -type f -name "full.corpus.zip" | while read zip_file; do
    unzip $zip_file -d $DOWNLOAD_DIR
done
mv "$DOWNLOAD_DIR/datasets/wikiconv-corpus/blocks.json" $DOWNLOAD_DIR