#!/bin/bash

python scripts/quality_dims/toxicity_annotate.py \
    --input_csv pefk.csv \
    --output_path output_datasets/pefk_toxicity.csv
    --api_key_file=perspective.key