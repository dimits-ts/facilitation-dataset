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
See [preprocessing.md](preprocessing.md).

## Facilitative detection
See [inference.md](inference.md).

## Acknowledgements

This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.