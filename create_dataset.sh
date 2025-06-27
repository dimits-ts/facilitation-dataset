mkdir -p "logs"
LOG_FILE="logs/pefk.log"
touch $LOG_FILE

cd scripts
bash master.sh ceri cmv_awry2 umod vmd wikitactics iq2 wikidisputes wikiconv | ts %Y-%m-%d_%H-%M-%S | tee "../$LOG_FILE"
