# The PEFK dataset

Repository housing the "Prosocial and Effective Facilitation in Konversations" (PEFK) dataset. This dataset is an aggregation and standardization of important facilitation datasets presented in Social Science literature. 

The dataset is provided as a large CSV file. Due to its overall size, it is not available directly on GitHub, but can be constructed by executing a shell script (see `Usage` Section).

The dataset is released under a CC-BY-SA License, and the code producing it uses the MIT software license.

**This repository is currently under development. We plan on adding more datasets and quantitative discussion quality metrics in the near future.**


## List of datasets used

- [WikiDisputes](https://aclanthology.org/2021.eacl-main.173/)
- [WikiTactics](https://arxiv.org/abs/2212.08353)
- [WikiConv](https://aclanthology.org/D18-1305/)  
- [Conversations Gone Awry / CMV II](https://arxiv.org/abs/1909.01362)
- [CeRI data](https://dl.acm.org/doi/10.1145/2307729.2307757)
- [User Moderation (UMOD)](https://aclanthology.org/2024.eacl-long.60/)
- [Virtual Moderation Dataset (VMD)](https://arxiv.org/abs/2503.16505)
- [Intelligence Squared 2 (IQ2)](https://aclanthology.org/N16-1017/)
- [Why How Who (WHoW)](https://aclanthology.org/2025.naacl-long.105/)
- [Fora](https://aclanthology.org/2024.acl-long.754/)

A list of references for each of the papers presenting the datasets can be found in the [refs.bib](refs.bib) file.

## Environment

The code that creates the dataset runs only on Linux (or WSL). We provide a conda environment with all dependencies in [`environment.yml`](environment.yml).

## Usage

```bash
git clone https://github.com/dimits-ts/facilitation-dataset.git

cd facilitation-dataset
conda env create -f environment.yml
conda activate pefk-dataset

bash create_dataset.sh
```

You may select a subset of the above datasets to be aggregated by deleting any of the following entries in the `create_dataset.sh` script:

```bash
bash master.sh wikiconv whow ceri cmv_awry2 umod vmd wikitactics iq2 fora | ts %Y-%m-%d_%H-%M-%S | tee "../$LOG_FILE"
``` 

## Important Notes

- The *Fora* dataset is NOT publicly available. Under an agreement with the MIT CCC we do not include this dataset by default in this repository, although the code to process it is present. 
    - If you have access to *Fora*, place the provided `.zip` file in the `project_root/downloads_external` directory.
    - You may request access to Fora following the researchers' [provided instructions](https://github.com/schropes/fora-corpus/blob/main/README.md)

- The WikiConv dataset is **extremely** large and may take multiple hours to download and process, depending on your hardware.


## Dataset Description

| Name        | Type   | Description                                                                 |
|-------------|--------|-----------------------------------------------------------------------------|
| conv_id     | string | The discussion's ID. Comments under the same discussion refer to the same discussion ID.|
| message_id  | string | The message's (comment's) unique ID.|
| reply_to    | string | The ID of the comment which the current comment responds to. nan if the comment does not respond to another comment (e.g., it's the Original Post (OP)). |
| user        | string | Username or hash of the user that posted the comment |
| is_moderator| bool   | Whether the user is a moderator/facilitator. In some datasets (e.g., UMOD, Wikitactics), normal users are considered facilitators if their *comments* are facilitative in nature. See Section `Preprocessing` for more details |
| moderation_supported | bool | True if the moderation labels are directly computed from the original dataset |
| escalated | bool | True if the comment caused a discussion to get derailed |
| escalation_supported | bool | True if the escalation labels are directly computed from the original dataset |
| text      | string | The contents of the comment  |
| dataset   | string | The dataset from which this comments originated from |
| notes     | JSON  | A dictionary holding notable dataset-specific information |  


## Preprocessing 

### General
- We exclude comments with no text

### Wikiconv
The Wikiconv corpus does not contain information about which user is a moderator/facilitator. Therefore, all comments relating to Wikiconv are tagged as non-moderators

- Additionally, we follow the [instructions of the original researchers](https://github.com/conversationai/wikidetox/blob/main/wikiconv/README.md), and select only discussions which have at least two comments by different users
    - Wikipedia (thankfully) does not track users who log in with only an IP address (in the original dataset, their user_id is always set to 0 and their username is of the form 211.111.111.xxx). We consider each such username to be a separate user.
    - Due to the size of the dataset, we have to partially load it during preprocessing. Thus, there is a small chance every 100,000 records that a discussion is marked as a false negative and a part of it gets discarded.
    - We only include English comments in the final dataset. We use a small, efficient library (`py3langid`) for language recognition, due to the large size of Wikiconv. We include every comment that is English with a confidence score of more than 75%. This can be trivially tuned in the `scripts/wikiconv.py` file. Non-english comments are discarded *before* selecting valid discussions (see point above).


### Wikitactics
In WikiTactics, we infer facilitative actions by whether the comment belongs in any of the following categories:
- Asking questions
- Coordinating edits
- Providing clarification
- Suggesting a compromise
- Contextualisation
- DH6: Refutation of opponent's argument (with evidence or reasoning)
- DH5: Counterargument with new evidence / reasoning
- DH7: Refuting the central point


### Wikidisputes
In WikiDisputes we mark a discussion as escalated when the derailement value is in the 60th upper percentile 

Since only 0.03% of the comments in the dataset are made by moderators, we mark the dataset as not supporting moderation and override its labels via classification.


### UMOD
In UMOD, facilitative actions are marked as a gradient from 0 (no facilitation) to 1 (full facilitation). We adopt a threshold of 0.75 to consider an action as facilitative, with more than 50% annotator agreement (measured as entropy in the original dataset).

### IQ2
In IQ2, we remove verbal linguistic markers ("...", "uh,").


## Facilitative detection

We use a Longformer model to estimate facilitative comments in the "Wikiconv" and "Conversations Gone Awry" datasets, by using the labels provided by the rest of the datasets.

We train two models; one on the entire labelled PEFK dataset (minus the VMD dataset), and one exclusively on the "WikiTactics" dataset. Our assumption was that the first model could leverage the multi-domain nature of PEFK (online/real-life/radio discussions) to extract generalizable facilitation features, while the second could leverage the same domain as Wikiconv to better annotate it.


### Results

|  Measure/Model | Combined  | Wikitactics-only  |   
|---|---|---|
| Accuracy |  0.78 |  0.5426 |  
| F1 (macro-weighted) |  0.69 |  0.5621 | 


### Training Time 
We used a single Quarto 6000 RTX GPU for training both models. 

|   | Combined  | Wikitactics-only  |   
|---|---|---|
| Training Time (hours) |  100 | 12 |  


### Replication code

Training code: [scripts/model_train.py](scripts/model_train.py). 

Evaluation code (on test set): [scripts/model_analysis.py](scripts/model_analysis.py). 

Inference code (on actual dataset): [scripts/model_inference.py](scripts/model_inference.py). 


### Parameters

We use the default pre-trained version of the Longformer model, with its weights frozen and a binary classification head. We use the default optimizer with Hugging Face Trainer's default parameters, but modify the BCE loss function to have a positive weight equal to the ratio of positive labels.

During training, we use a batch size of 32, max sequence length of 4096, and pin all seeds to the number 42. The wikitactics/combined models were left to run with an Early Stopping algorithm for 100/1000 epochs respectively, with delta=10e-5 and a patience of 20, with evaluation every 100/400 steps. We select the best model according to f1 scores on the validation set. We stopped the training of the combined model after 46,000 steps.



## Acknowledgements

This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.