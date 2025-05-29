DOWNLOAD_DIR="downloads/wikiconv"
URL="https://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/"

mkdir -p $DOWNLOAD_DIR
wget -r -nH --no-parent --cut-dirs=1 -P $DOWNLOAD_DIR $URL -A .json .zip
