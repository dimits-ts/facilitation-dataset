## Taxonomy Annotation

### Detecting facilitative comments
We consider a comment to be facilitative if it is either marked as such in the original dataset, or if our classifier believes it to be with a confidence of more than $60\%$.

Literature has shown that comments not made by facilitators can still be facilitative, which is why we include comments not necessarily made by facilitators, in datasets where they are clearly labelled as such.


### Sampling
Since the wikidisputes, wikitactics and wikiconv datasets represent the same population, we consider them the same dataset in this analysis ("wikipedia").

Since the sizes of our datasets do not reflect real-world population sizes, we select the same number of facilitative comments from each dataset. Thus, a maximum of $2000$ comments is drawn from each dataset. Some datasets have a lower count of comments bringing the total of comments to $12864$. The inclusion of the $60\%$ threshold mitigates this imbalance.


### Classification

We consider the bottom-level tactics for each of the $4$ taxonomies considered (see [the taxonomy file](llm_classification/taxonomy.yaml)). We then run a binary classification task using an open-source LLM for each of these $33$ tactics.

For each tactic we include its name, a description and example provided by the papers introducing them. We also provide the comment to be classified and $2$ preceding comments as context. Each comment is allowed a maximum of $2000$ characters due to VRAM concerns.

### Technical Details
We use a LLaMa3.3-70b instruction-tuned model quantized to 4 bits. The instruction prompt can be found [in this file](llm_classification/prompt.txt).

Execution time was $XX$ days. The experiments were run on 2 Quadro RTX GPUs.
