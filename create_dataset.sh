LOG_FILE="logfile.log"
touch $LOG_FILE

cd scripts
bash master.sh ceri cmv_awry2 umod vmd wikiconv wikitactics | ts %Y-%m-%d_%H-%M-%S | tee "../$LOG_FILE"