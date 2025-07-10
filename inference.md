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