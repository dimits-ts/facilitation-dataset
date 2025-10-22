# Facilitator detection

We use a [ModernBERT](https://arxiv.org/abs/2412.13663) model to estimate facilitative comments in the "Wikiconv", "Wikidisputes" and "Conversations Gone Awry" datasets, by using the labels provided by the rest of the datasets. The classifier is trained by the rest of the *non-synthetic* datasets.

The comments are given to the model in XML format. We use the tags \<CTX\> for preceding comments (context) and \<TRT\> for the actual (target) comments to be classified. These tokens are not added to the tokenizer as special tokens, to keep implementation simple.

We use a maximum of 2 comments for each training sample, and each of these comments is truncated to 5000 characters (tags remain no matter what). We do not use information about users, since most usernames are hashed during dataset preprocessing.

Example:
```
<CTX> hello! <\CTX>
<CTX> shut up no one loves you <\CTX>
<TRT> not cool man <\TRT>
```

## Facilitator comments vs facilitative comments

The detection between comments made by a facilitator vs. the comments that are facilitative in nature may sound trivial or academic in practice. However, this couldn't be further from the truth. 

Consider the following example:

```
<CTX> Maybe we should start by collecting examples of unclear moderation cases before suggesting new rules.<\CTX>
<TRT> Yeah, OK <\TRT>
```

And contrast it with the following:
```
<CTX> I think we should ban users who repeatedly downvote others without reason<\CTX>
<TRT> Yeah, OK <\TRT>
```
The first target comment is obviously facilitative, and the second is not, even though the utterances are the same. However, both could have been reasonably be made *by the same person*, perhaps even in the same discussion. If that person is a facililator, both comments will be considered "facilitator comments", even though their contents may not actually be facilitative. 

With the exception of UMOD, all datasets describing facilitation in PEFK only consider whether the speaker is a facilitator. This may be a useful heuristic (and indeed, the only heuristic in the absense of large scale annotation in the style of UMOD), but it inherently introduces a lot of noise on whether a comment actually facilitates discussion. The latter is left to future work. 

We note that in the case of formal moderation/facilitation datasets, specially trained moderators will most frequently intervene in a facilitative way, meaning that the study of facilitator comments remains useful.

## Reconciling different domains

In terms of lexical contents, the PEFK dataset encompasses two broad but distinct categories; written comments and transcribed speech. While both share the fundamental properties of English, they vary greatly in composition and turn taking (e.g., there can be no interruptions in written speech). This distinction may also be present for the facilitation detection task, since online and real-life facilitation styles can vary greately.

As such, we train two distinct models for each of the two categories. We also train a model trained on the entire dataset, for comparison.

## Training Time

We used a single Quarto 6000 RTX GPU for training. Training time was 3 hours, 19 hours and 26 hours for the "written", "spoken" and "all" classifiers respectively. Inference took ?? hours.


## Replication code

- [Training code](../scripts/facilitation_train.py)

- [Inference code](../scripts/facilitation_inference.py)

- [Run configurations](../augment_dataset.sh)


## Hyperparameters

We use the default pre-trained version of the ModernBERT-large model, with its weights frozen and a binary classification head. We use the default optimizer with Hugging Face Trainer's default parameters, but modify the BCE loss function to have a positive weight equal to the ratio of positive labels. We also use bucketing (creating batches with examples that have similar sizes) to increase efficiency.

For more information, we recommend checking out the constants used in the [training code](../scripts/facilitation_train.py).


## Notes

In an earlier version of our experiments, we had accidentally considered argumentative tactics as facilitation in the WIkitactics dataset. When correcting this mistake, we noted a very large improvement in f1 scores (>0.1), meaning that the tactics we selected had much more in common with faciltiative comments from other datasets.


## Training Results

Keep in mind that the dataset performance statistics are applied on the test set, unlike the precision-recall curves which are applied on the validation set. This is necessary, since we use the pr-curves for thresholding facilitator comments in subsequent experiments.

### Spoken Classifier

#### Dataset Performance

| Dataset  | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| fora     | 0.5372    | 0.6477 | 0.5873 | -       |
| iq2      | 0.5908    | 0.7136 | 0.6464 | -       |
| whow     | 0.5871    | 0.6895 | 0.6342 | -       |


#### Threshold vs Precision, Recall, F1

| Threshold | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| 0         | 0.3564    | 1.0    | 0.5255 |
| 0.05      | 0.3577    | 0.9997 | 0.5268 |
| 0.1       | 0.3649    | 0.9961 | 0.5341 |
| 0.15      | 0.3784    | 0.9874 | 0.5471 |
| 0.2       | 0.3975    | 0.9719 | 0.5642 |
| 0.25      | 0.4219    | 0.9497 | 0.5843 |
| 0.3       | 0.4488    | 0.914  | 0.602  |
| 0.35      | 0.4778    | 0.8694 | 0.6167 |
| 0.4       | 0.5065    | 0.8166 | 0.6252 |
| 0.45      | 0.5333    | 0.7525 | 0.6243 |
| 0.5       | 0.5658    | 0.684  | 0.6193 |
| 0.55      | 0.5986    | 0.6154 | 0.6069 |
| 0.6       | 0.6273    | 0.539  | 0.5798 |
| 0.65      | 0.6578    | 0.4531 | 0.5366 |
| 0.7       | 0.6966    | 0.3753 | 0.4878 |
| 0.75      | 0.7269    | 0.2916 | 0.4162 |
| 0.8       | 0.7485    | 0.209  | 0.3267 |
| 0.85      | 0.7676    | 0.1317 | 0.2249 |
| 0.9       | 0.8014    | 0.0635 | 0.1176 |
| 0.95      | 0.8551    | 0.0166 | 0.0325 |
| 1.0       | 0.0       | 0.0    | 0.0    |


---

### Written Classifier

#### Dataset Performance

| Dataset      | Precision | Recall | F1    | Support |
|--------------|-----------|--------|-------|---------|
| ceri         | 0.5       | 0.8082 | 0.6178 | -       |
| umod         | 0.3947    | 0.5    | 0.4412 | -       |
| wikitactics  | 0.4735    | 0.8562 | 0.6098 | -       |

#### Threshold vs Precision, Recall, F1

| Threshold | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| 0         | 0.2586    | 1.0    | 0.4109 |
| 0.05      | 0.2854    | 0.996  | 0.4436 |
| 0.1       | 0.3095    | 0.992  | 0.4718 |
| 0.15      | 0.3275    | 0.9719 | 0.4899 |
| 0.2       | 0.346     | 0.9518 | 0.5075 |
| 0.25      | 0.3648    | 0.9317 | 0.5243 |
| 0.3       | 0.3802    | 0.9116 | 0.5366 |
| 0.35      | 0.4048    | 0.8795 | 0.5544 |
| 0.4       | 0.4206    | 0.8514 | 0.5631 |
| 0.45      | 0.4433    | 0.8313 | 0.5782 |
| 0.5       | 0.4685    | 0.8072 | 0.5929 |
| 0.55      | 0.5067    | 0.755  | 0.6065 |
| 0.6       | 0.5304    | 0.6667 | 0.5907 |
| 0.65      | 0.572     | 0.6225 | 0.5962 |
| 0.7       | 0.6079    | 0.5542 | 0.5798 |
| 0.75      | 0.6505    | 0.4859 | 0.5563 |
| 0.8       | 0.6818    | 0.4217 | 0.5211 |
| 0.85      | 0.7311    | 0.3494 | 0.4728 |
| 0.9       | 0.7632    | 0.2329 | 0.3569 |
| 0.95      | 0.7907    | 0.1365 | 0.2329 |
| 1.0       | 0.0       | 0.0    | 0.0    |


---

### All (Written + Spoken)

#### Dataset Performance

| dataset     | precision | recall  | f1        | support |
|--------------|------------|----------|-----------|----------|
| ceri         | 0.630952   | 0.602273 | 0.616279  |          |
| fora         | 0.518129   | 0.647188 | 0.575512  |          |
| iq2          | 0.592543   | 0.702447 | 0.642831  |          |
| umod         | 0.083333   | 0.045455 | 0.058824  |          |
| whow         | 0.583186   | 0.711663 | 0.641051  |          |
| wikitactics  | 0.472222   | 0.615942 | 0.534591  |          |


#### Threshold vs Precision, Recall, F1

| Threshold | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| 0         | 0.3478    | 1.0    | 0.5161 |
| 0.05      | 0.3506    | 0.9989 | 0.519  |
| 0.1       | 0.3617    | 0.9958 | 0.5307 |
| 0.15      | 0.377     | 0.9858 | 0.5454 |
| 0.2       | 0.399     | 0.9716 | 0.5657 |
| 0.25      | 0.4203    | 0.9438 | 0.5816 |
| 0.3       | 0.4485    | 0.9105 | 0.601  |
| 0.35      | 0.4748    | 0.864  | 0.6128 |
| 0.4       | 0.5059    | 0.812  | 0.6234 |
| 0.45      | 0.5321    | 0.7516 | 0.6231 |
| 0.5       | 0.5636    | 0.6839 | 0.618  |
| 0.55      | 0.5932    | 0.613  | 0.603  |
| 0.6       | 0.6303    | 0.5434 | 0.5837 |
| 0.65      | 0.6486    | 0.4544 | 0.5344 |
| 0.7       | 0.6811    | 0.3668 | 0.4768 |
| 0.75      | 0.7133    | 0.2835 | 0.4058 |
| 0.8       | 0.7623    | 0.2037 | 0.3215 |
| 0.85      | 0.7743    | 0.1171 | 0.2034 |
| 0.9       | 0.8136    | 0.0504 | 0.0949 |
| 0.95      | 1.0       | 0.0095 | 0.0187 |
| 1.0       | 0.0       | 0.0    | 0.0    |

