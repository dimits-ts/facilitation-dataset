#!/bin/bash
DOWNLOAD_DIR="../downloads/ceri"
DOWNLOAD_PATH="../downloads/ceri/ceri.xlsx"
URL="https://scholarship.law.cornell.edu/context/ceri/article/1019/type/native/viewcontent"

mkdir -p $DOWNLOAD_DIR
wget --no-verbose -nc -O $DOWNLOAD_PATH $URL