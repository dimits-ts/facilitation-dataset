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
- [UMOD](https://aclanthology.org/2024.eacl-long.60/)
- [VMD](https://arxiv.org/abs/2503.16505)

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

You may select a subset of the above datasets to be aggregated by deleting any of the following entries in the `scripts/create_dataset.sh` file:

```bash
declare -a datasetnames=("ceri" "cmv_awry2" "umod" "vmd" "wikidisputes" "wikitactics")
``` 

**Warning:** The WikiConv dataset is **extremely** large and may take multiple hours/days to download and process, depending on your hardware. It is recommended to initially skip this dataset.

## Dataset Description


| Name        | Type   | Description                                                                 |
|-------------|--------|-----------------------------------------------------------------------------|
| conv_id     | string | The discussion's ID. Comments under the same discussion refer to the same discussion ID.|
| message_id  | string | The message's (comment's) unique ID.|
| reply_to    | string | The ID of the comment which the current comment responds to. nan if the comment does not respond to another comment (e.g., it's the Original Post (OP)). |
| user        | string | Username or hash of the user that posted the comment |
| is_moderator| bool   | Whether the user is a moderator/facilitator. In some datasets (e.g., UMOD), normal users are considered facilitators if their comments are facilitative in nature. See Section `Preprocessing` for more details |
| text      | string | The contents of the comment  |
| dataset   | string | The dataset from which this comments originated from |
| notes     | JSON  | A dictionary holding notable dataset-specific information |  


## Preprocessing 

- We exclude comments with no text
- The Wikiconv corpus does not contain information about which user is a moderator/facilitator. Therefore, all comments relating to Wikiconv are tagged as non-moderators
- In WikiDisputes, we infer facilitative actions by whether the comment belongs in any of the following categories:
    - Asking questions
    - Coordinating edits
    - Providing clarification
    - Suggesting a compromise
    - Contextualisation
    - DH6: Refutation of opponent's argument (with evidence or reasoning)
    - DH5: Counterargument with new evidence / reasoning
    - DH7: Refuting the central point
- In UMOD, facilitative actions are marked as a gradient from 0 (no facilitation) to 1 (full facilitation). We adopt a threshold of 0.75 to consider an action as facilitative, with more than 50% annotator agreement (measured as entropy in the original dataset).


## Acknowledgement

This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.