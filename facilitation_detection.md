## Facilitative detection

We use a [ModernBERT](https://arxiv.org/abs/2412.13663) model to estimate facilitative comments in the "Wikiconv", "Wikidisputes" and "Conversations Gone Awry" datasets, by using the labels provided by the rest of the datasets. The classifier is trained by the rest of the *non-synthetic* datasets.

The comments are given to the model in XML format. We use the tags \<CTX\> for preceding comments (context), \<USR\> tags to denote the usernames who posted each comment, and \<TRT\> for the actual (target) comments to be classified. We use a maximum of 4 comments for each training sample, although this number is reduced should it not fit to the model's context window of 8192 tokens. We do not truncate the input. These tokens are not added to the tokenizer as special tokens, to keep implementation simple.

### Results

| Dataset     |   Loss   | Accuracy |   F1     |
|-------------|----------|----------|----------|
| ALL         | 0.473968 | 0.774125 | 0.703064 |
| ceri        | 0.340454 | 0.869333 | 0.723164 |
| fora        | 0.525332 | 0.741777 | 0.664920 |
| iq2         | 0.427206 | 0.802003 | 0.748461 |
| umod        | 0.488729 | 0.783069 | 0.226415 |
| whow        | 0.444721 | 0.795395 | 0.745543 |
| wikitactics | 0.671555 | 0.621212 | 0.471831 |


### Training Time

We used a single Quarto 6000 RTX GPU for training. The inference took 46 hours, while training took about 3 hours.


### Replication code

Training code: [scripts/model_train.py](scripts/model_train.py). 

Inference code: [scripts/model_inference.py](scripts/model_inference.py). 


### Parameters

We use the default pre-trained version of the ModernBERT model, with its weights frozen and a binary classification head. We use the default optimizer with Hugging Face Trainer's default parameters, but modify the BCE loss function to have a positive weight equal to the ratio of positive labels. We also use bucketing (creating batches with examples that have similar sizes) to increase efficiency.

During training, we use a batch size of 32, max sequence length of 8192, and pin all seeds to the number 42. We ran the models on 120 epochs with Early Stopping, with warmup of 12000 steps, delta=10e-5 and a patience of 5, with evaluation every 4000 steps. We select the best model according to the evaluation loss.