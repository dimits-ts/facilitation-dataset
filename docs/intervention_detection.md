## Intervention Detection

Since the escalation labels provided by the datasets are given only on the discussion level, and not the comment level, we can not use them to determine when a moderator should intervene. 

The next best case would be to assume human judgements are correct and define gold labels for interventions as the comments before the actual moderators spoke. This assumes that (a) humans know when to intervene (which is not supported by literature) and (b) that we are not missing any cases where an intervention should have happened but didn't (to the detriment of the discussion). Neither of these assumptions generally hold in our dataset, and thus the judgements of our model should be taken with a grain of salt. Nevertheless, they provide a useful heuristic/baseline.

We follow the same methodology and code as in [the facilitation detection task](facilitation_detection.md).


### Results (test set)

| Dataset     | Loss     | Accuracy | F1 Score |
| ----------- | -------- | -------- | -------- |
| **ALL**     | 0.631963 | 0.636788 | 0.541598 |
| ceri        | 0.610345 | 0.688822 | 0.279720 |
| fora        | 0.645520 | 0.630862 | 0.525292 |
| iq2         | 0.623696 | 0.640565 | 0.563461 |
| umod        | 0.656473 | 0.619565 | 0.186047 |
| whow        | 0.612166 | 0.647949 | 0.580645 |
| wikitactics | 0.705838 | 0.556901 | 0.411576 |


### PR curves (validation set)

| Threshold | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| 0.00      | 0.3485    | 1.0000 | 0.5168   |
| 0.05      | 0.3485    | 1.0000 | 0.5168   |
| 0.10      | 0.3488    | 0.9999 | 0.5172   |
| 0.15      | 0.3520    | 0.9980 | 0.5205   |
| 0.20      | 0.3621    | 0.9908 | 0.5303   |
| 0.25      | 0.3738    | 0.9780 | 0.5409   |
| 0.30      | 0.3878    | 0.9551 | 0.5516   |
| 0.35      | 0.4051    | 0.9112 | 0.5609   |
| 0.40      | 0.4270    | 0.8456 | 0.5675   |
| 0.45      | 0.4516    | 0.7490 | 0.5635   |
| 0.50      | 0.4783    | 0.6227 | 0.5410   |
| 0.55      | 0.5145    | 0.4928 | 0.5034   |
| 0.60      | 0.5512    | 0.3492 | 0.4275   |
| 0.65      | 0.5908    | 0.2211 | 0.3218   |
| 0.70      | 0.6217    | 0.1174 | 0.1975   |
| 0.75      | 0.6315    | 0.0537 | 0.0990   |
| 0.80      | 0.6606    | 0.0243 | 0.0468   |
| 0.85      | 0.5833    | 0.0056 | 0.0111   |
| 0.90      | 0.6250    | 0.0007 | 0.0013   |
| 0.95      | 0.0000    | 0.0000 | 0.0000   |
| 1.00      | 0.0000    | 0.0000 | 0.0000   |
