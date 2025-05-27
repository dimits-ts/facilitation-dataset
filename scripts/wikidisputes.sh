DATA_DIR="./datasets"
DOWNLOAD_DIR="./downloads"
DATA_PATH="$DATA_DIR/wikidisputes/"
DOWNLOAD_PATH="$DOWNLOAD_DIR/wikidisputes.tar.gz"

mkdir -p $DATA_PATH
wget -nc -O $DOWNLOAD_PATH "https://github.com/christinedekock11/wikidisputes/raw/refs/heads/main/data.tar.gz"
tar -xvzf $DOWNLOAD_PATH -C $DATA_PATH