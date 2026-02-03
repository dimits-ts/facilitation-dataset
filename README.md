# The PEFK dataset

The "Prosocial and Effective Facilitation in Konversations" (PEFK) dataset is an aggregation and standardization of important facilitation datasets presented in Social Science literature. It includes numerous metrics and taxonomy labels from Machine Learning, Deep Learning and LLM classifiers.

The dataset will be provided as a single file upon the completion of the project. The current version can be constructed by executing a shell script (see `Usage` Section). It is released under a CC-BY-SA License, and the code producing it uses the GPLv3 software license.

**This repository is currently under development. We plan on adding more datasets and quantitative discussion quality metrics in the near future.**


## List of datasets used

| Name                             | Size (#comments) | Domain                   | Format      | Link                                                   |
| -------------------------------- | ---------------- | ------------------------ | ----------- | ------------------------------------------------------ |
| WikiDisputes                     | 96,320           | Forum                    | Text   | [Link](https://aclanthology.org/2021.eacl-main.173/)   |
| WikiTactics                      | 3,850            | Forum                    | Text    | [Link](https://arxiv.org/abs/2212.08353)               |
| WikiConv                         | 17,806,373       | Forum                    | Text    | [Link](https://aclanthology.org/D18-1305/)             |
| Conversations Gone Awry / CMV II | 40,607           | Forum                    | Text    | [Link](https://arxiv.org/abs/1909.01362)               |
| CeRI data                        | 3,700            | Forum                    | Text    | [Link](https://dl.acm.org/doi/10.1145/2307729.2307757) |
| User Moderation (UMOD)           | 2,000            | Forum                    | Text    | [Link](https://aclanthology.org/2024.eacl-long.60/)    |
| Virtual Moderation Dataset (VMD) | 3,563            | Forum                    | Text    | [Link](https://arxiv.org/abs/2503.16505)               |
| Intelligence Squared 2 (IQ2)     | 34,245           | Debate       | Oral-Transcribed | [Link](https://aclanthology.org/N16-1017/)             |
| Why How Who (WHoW)               | 25,542           | Radio / TV               | Oral-Transcribed | [Link](https://aclanthology.org/2025.naacl-long.105/)  |
| Fora                             | 39,438           | Deliberation | Oral-Transcribed | [Link](https://aclanthology.org/2024.acl-long.754/)    |


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

bash create_base_dataset.sh wikiconv whow ceri cmv_awry2 umod vmd wikitactics iq2 fora
```

## Important Notes

- The *Fora* dataset is NOT publicly available. Under an agreement with the MIT CCC we do not include this dataset by default in this repository, although the code to process it is present. 
    - If you have access to *Fora*, place the provided `.zip` file in the `<project_root>/downloads_external` directory.
    - You may request access to Fora following the researchers' [provided instructions](https://github.com/schropes/fora-corpus/blob/main/README.md)

- The WikiConv dataset is **extremely** large and may take multiple hours to download and process, depending on your hardware.


## Dataset Description

| Name        | Type   | Description  |
|-------------|--------|-----------------------------------------------------------------------------|
| conv_id     | string | The discussion's ID. Comments under the same discussion refer to the same discussion ID.|
| message_id  | string | The message's (comment's) unique ID.|
| reply_to    | string | The ID of the comment which the current comment responds to. nan if the comment does not respond to another comment (e.g., it's the Original Post (OP)). |
| user        | string | Username or hash of the user that posted the comment |
| is_moderator| bool   | Whether the user is a moderator/facilitator. In some datasets (e.g., UMOD, Wikitactics), normal users are considered facilitators if their *comments* are facilitative in nature. See Section `Preprocessing` for more details |
| moderation_supported | bool | True if the moderation labels are directly computed from the original dataset |
| escalated | bool | A discussion-level measure denoting discussions which have been derailed |
| escalation_supported | bool | True if the escalation labels are directly computed from the original dataset |
| text      | string | The contents of the comment  |
| dataset   | string | The dataset from which this comments originated from |
| notes     | JSON  | A dictionary holding notable dataset-specific information |
| toxicity | float | The "toxicity" score given to the comment by the Perspective API | 
| severe_toxicity | float | The "severe toxicity" score given to the comment by the Perspective API |


## Preprocessing 

### General
- We exclude comments with no text
- We exclude discussions with less than two distinct participants
- We exclude discussions which are common between wikitactics and wikiconv as well as wikidisputes and wikiconv. 
    - There may be duplicate discussions between wikidisputes and wikitactics, but we allow them since they feature complementary information

### Wikiconv
The Wikiconv corpus does not contain information about which user is a moderator/facilitator. Therefore, all comments relating to Wikiconv are tagged as non-moderators

- Additionally, we follow the [instructions of the original researchers](https://github.com/conversationai/wikidetox/blob/main/wikiconv/README.md), and select only discussions which have at least two comments by different users
    - Wikipedia (thankfully) does not track users who log in with only an IP address (in the original dataset, their user_id is always set to 0 and their username is of the form 211.111.111.xxx). We consider each such username to be a separate user.
    - Due to the size of the dataset, we have to partially load it during preprocessing. Thus, there is a small chance every 100,000 records that a discussion is marked as a false negative and a part of it gets discarded.
    - We only include English comments in the final dataset. We use a small, efficient library (`py3langid`) for language recognition, due to the large size of Wikiconv. Non-english comments are discarded *before* selecting valid discussions (see point above).


### Wikitactics
We infer facilitative actions by whether the comment belongs in any of the following categories:
- Asking questions
- Coordinating edits
- Providing clarification
- Suggesting a compromise
- Contextualisation

The above tactics are a subset of the Coordinative labels used in the WikiTactics paper. They were selected because they are not used neccesarily on 1-1 discussions; they could reasonably be applied by third-party participants. Contrast them with other Coordinative labels such as "Conceding/recanting" and "I don't know".

### Wikidisputes
Since only 0.03% of the comments in the dataset are made by moderators, we mark the dataset as not supporting moderation.


### UMOD
Facilitative actions are marked as a gradient from 0 (no facilitation) to 1 (full facilitation). We adopt a threshold of 0.75 to consider an action as facilitative, with more than 50% annotator agreement (measured as entropy in the original dataset).


### CMV-AWRY2
We mark a discussion as escalated when the derailement value (from the official dataset) is in the 60th upper percentile.

We remove deleted ("[deleted]") comments.


## Acknowledgements

This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.