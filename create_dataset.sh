LOG_FILE="logfile.log"
touch $LOG_FILE

cd scripts
bash master.sh | ts %Y-%m-%d_%H-%M-%S | tee "../$LOG_FILE"