## Facilitative detection

We use a [ModernBERT](https://arxiv.org/abs/2412.13663) model to estimate facilitative comments in the "Wikiconv", "Wikidisputes" and "Conversations Gone Awry" datasets, by using the labels provided by the rest of the datasets. The classifier is trained by the rest of the *non-synthetic* datasets.

The comments are given to the model in XML format. We use the tags \<CTX\> for preceding comments (context) and \<TRT\> for the actual (target) comments to be classified. These tokens are not added to the tokenizer as special tokens, to keep implementation simple.

We use a maximum of 2 comments for each training sample, and each of these comments is truncated to 5000 characters (tags remain no matter what). We do not use information about users, since most usernames are hashed during dataset preprocessing.

Example:
```
<CTX> hello! <\CTX>
<CTX> shut up no one loves you <\CTX>
<TRT> not cool man <\TRT>
```

### Results (test set)

| Dataset     | Loss     | Accuracy | F1 Score |
| ----------- | -------- | -------- | -------- |
| **ALL**     | 0.374406 | 0.835356 | 0.767130 |
| ceri        | 0.309790 | 0.883469 | 0.757062 |
| fora        | 0.543159 | 0.734203 | 0.621429 |
| iq2         | 0.274744 | 0.906324 | 0.874852 |
| umod        | 0.374616 | 0.868545 | 0.263158 |
| whow        | 0.184099 | 0.934982 | 0.911557 |
| wikitactics | 0.781466 | 0.568345 | 0.400000 |

### PR curves (validation set)

| Threshold | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| 0.00      | 0.3493    | 1.0000 | 0.5177   |
| 0.05      | 0.3955    | 0.9949 | 0.5660   |
| 0.10      | 0.4367    | 0.9837 | 0.6049   |
| 0.15      | 0.4766    | 0.9674 | 0.6386   |
| 0.20      | 0.5159    | 0.9478 | 0.6682   |
| 0.25      | 0.5541    | 0.9267 | 0.6935   |
| 0.30      | 0.5929    | 0.9003 | 0.7150   |
| 0.35      | 0.6304    | 0.8672 | 0.7301   |
| 0.40      | 0.6680    | 0.8375 | 0.7432   |
| 0.45      | 0.7043    | 0.8047 | 0.7512   |
| 0.50      | 0.7419    | 0.7700 | 0.7557   |
| 0.55      | 0.7762    | 0.7340 | 0.7545   |
| 0.60      | 0.8043    | 0.6921 | 0.7440   |
| 0.65      | 0.8333    | 0.6488 | 0.7296   |
| 0.70      | 0.8622    | 0.6067 | 0.7122   |
| 0.75      | 0.8887    | 0.5567 | 0.6845   |
| 0.80      | 0.9220    | 0.5055 | 0.6530   |
| 0.85      | 0.9476    | 0.4474 | 0.6078   |
| 0.90      | 0.9712    | 0.3728 | 0.5388   |
| 0.95      | 0.9886    | 0.2737 | 0.4287   |
| 1.00      | 0.0000    | 0.0000 | 0.0000   |



### Training Time

We used a single Quarto 6000 RTX GPU for training. The inference took 46 hours, while training took about 3 hours.


### Replication code

Training code: [scripts/model_train.py](scripts/model_train.py). 

Inference code: [scripts/model_inference.py](scripts/model_inference.py). Note that we do use truncation during inference.


### Parameters

We use the default pre-trained version of the ModernBERT-large model, with its weights frozen and a binary classification head. We use the default optimizer with Hugging Face Trainer's default parameters, but modify the BCE loss function to have a positive weight equal to the ratio of positive labels. We also use bucketing (creating batches with examples that have similar sizes) to increase efficiency.

During training, we use a batch size of 32, max sequence length of 8192, and pin all seeds to the number 42. We ran the models on 120 epochs with Early Stopping, with warmup of 12000 steps, delta=10e-5 and a patience of 5, with evaluation every 4000 steps. We select the best model according to the evaluation loss.


### Notes

In an earlier version of our experiments, we had accidentally considered argumentative tactics as facilitation in the WIkitactics dataset. When correcting this mistake, we noted a very large improvement in f1 scores (>0.1), meaning that the tactics we selected had much more in common with faciltiative comments from other datasets.