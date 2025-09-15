#!/bin/bash
bash create_base_dataset.sh

LOG_FILE="logs/pefk_augmented.log"
cd scripts
bash run_classifiers.sh | tee "../$LOG_FILE"
