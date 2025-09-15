# The PEFK dataset

The"Prosocial and Effective Facilitation in Konversations" (PEFK) dataset is an aggregation and standardization of important facilitation datasets presented in Social Science literature. It includes numerous metrics and taxonomy labels from Machine Learning, Deep Learning and LLM classifiers.

The dataset will be provided as a single file upon the completion of the project. The current version can be constructed by executing a shell script (see `Usage` Section). It is released under a CC-BY-SA License, and the code producing it uses the MIT software license.

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

A list of bibliographical references for each of the respective papers can be found in the [refs.bib](refs.bib) file.


## Environment

The code that creates the dataset runs on any Linux environment. Other OS environments are not supported.

We provide a conda environment with all dependencies in [`environment.yml`](environment.yml). See `Usage` for more information.

## Usage

```bash
git clone https://github.com/dimits-ts/facilitation-dataset.git

cd facilitation-dataset
conda env create -f environment.yml
conda activate pefk-dataset

# data only contained inside the datasets
bash create_base_dataset.sh  

# uncomment to include ``Inferred'' data (see Table below)
# bash create_augmented_dataset.sh 

# uncomment to add taxonomy information to the dataset
# bash analyze_taxonomies.bash 
```

## Important Notes

- The *Fora* dataset is NOT publicly available. Under an agreement with the MIT CCC we do not include this dataset by default in this repository, although the code to process it is present. 
    - If you have access to *Fora*, place the provided `.zip` file in the `project_root/downloads_external` directory.
    - You may request access to Fora following the researchers' [provided instructions](https://github.com/schropes/fora-corpus/blob/main/README.md)

- The WikiConv dataset is **extremely** large and may take multiple hours to download and process, depending on your hardware.


## Dataset Description

| Name        | Type   | Description  | Inferred |
|-------------|--------|-----------------------------------------------------------------------------| --------|
| conv_id     | string | The discussion's ID. Comments under the same discussion refer to the same discussion ID.| |
| message_id  | string | The message's (comment's) unique ID.| |
| reply_to    | string | The ID of the comment which the current comment responds to. nan if the comment does not respond to another comment (e.g., it's the Original Post (OP)). | |
| user        | string | Username or hash of the user that posted the comment | |
| is_moderator| bool   | Whether the user is a moderator/facilitator. In some datasets (e.g., UMOD, Wikitactics), normal users are considered facilitators if their *comments* are facilitative in nature. See Section `Preprocessing` for more details ||
| moderation_supported | bool | True if the moderation labels are directly computed from the original dataset | |
| escalated | bool | A discussion-level measure denoting discussions which have been derailed | |
| escalation_supported | bool | True if the escalation labels are directly computed from the original dataset | |
| text      | string | The contents of the comment  | |
| dataset   | string | The dataset from which this comments originated from | |
| notes     | JSON  | A dictionary holding notable dataset-specific information | |
| toxicity | float | The "toxicity" score given to the comment by the Perspective API | ✔ |
| severe_toxicity | float | The "severe toxicity" score given to the comment by the Perspective API | ✔ |
| mod_probabilities | float | The probability that the comment is facilitative (given by a DL classifier - see Section "Facilitative detection") | ✔ |
| should_have_intervened | bool | Whether the  next comment is facilitative. Valid only where moderation_supported=1 | ✔ |
| should_have_intervened_probabilities | float | The probability that the *next* comment is facilitative (given by a DL classifier - see Section "Facilitative detection") | ✔ |

Inferred columns contain information not obtained by the actual datasets, but by our own analysis.


## Documentation

* [Preprocessing](docs/preprocessing.md)

* [Detecting facilitative comments](docs/facilitation_detection.md)

* [Predicting when facilitators intervene](docs/intervention_detection.md)

* [Analyzing facilitative taxonomies](docs/taxonomy_annotation.md)


## Acknowledgements

This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.