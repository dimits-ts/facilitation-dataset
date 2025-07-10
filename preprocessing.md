## Preprocessing 

### General
- We exclude comments with no text
- We remove discussions with less than two distinct participants

### Wikiconv
The Wikiconv corpus does not contain information about which user is a moderator/facilitator. Therefore, all comments relating to Wikiconv are tagged as non-moderators

- Additionally, we follow the [instructions of the original researchers](https://github.com/conversationai/wikidetox/blob/main/wikiconv/README.md), and select only discussions which have at least two comments by different users
    - Wikipedia (thankfully) does not track users who log in with only an IP address (in the original dataset, their user_id is always set to 0 and their username is of the form 211.111.111.xxx). We consider each such username to be a separate user.
    - Due to the size of the dataset, we have to partially load it during preprocessing. Thus, there is a small chance every 100,000 records that a discussion is marked as a false negative and a part of it gets discarded.
    - We only include English comments in the final dataset. We use a small, efficient library (`py3langid`) for language recognition, due to the large size of Wikiconv. We include every comment that is English with a confidence score of more than 75%. This can be trivially tuned in the `scripts/wikiconv.py` file. Non-english comments are discarded *before* selecting valid discussions (see point above).


### Wikitactics
We infer facilitative actions by whether the comment belongs in any of the following categories:
- Asking questions
- Coordinating edits
- Providing clarification
- Suggesting a compromise
- Contextualisation
- DH6: Refutation of opponent's argument (with evidence or reasoning)
- DH5: Counterargument with new evidence / reasoning
- DH7: Refuting the central point


### Wikidisputes
Since only 0.03% of the comments in the dataset are made by moderators, we mark the dataset as not supporting moderation and override its labels via classification.


### UMOD
Facilitative actions are marked as a gradient from 0 (no facilitation) to 1 (full facilitation). We adopt a threshold of 0.75 to consider an action as facilitative, with more than 50% annotator agreement (measured as entropy in the original dataset).

### IQ2
We remove verbal linguistic markers ("...", "uh,").

### CMV-AWRY2
We mark a discussion as escalated when the derailement value (from the official dataset) is in the 60th upper percentile.

We remove deleted ("[deleted]") comments.