# Hindi-Sentence-Completion
Supplementary code for the work "Local context models for the clause-final verb prediction in Hindi"

# Data & tools
1. The main corpus data is taken from [here](http://www.cfilt.iitb.ac.in/iitb_parallel/). This is used to calculate the HDMI values and do the further analysis with distance and word-order. 

2. To get the animacy annotations of the nouns, we extracted the annotated nouns collected in the data used in this [work](https://www.aclweb.org/anthology/W13-2320/). 

3. The parser used in this work is taken from [here](https://bitbucket.org/iscnlp/workspace/projects/ISCNLP).

# Requirements

Other than the parser mentioned in the above section, the code requires to install:

1. Mongo database ([mongodb](https://docs.mongodb.com/manual/installation/), [pymongo](https://docs.opsmanager.mongodb.com/v1.2/monitoring/tutorial/install-pymongo/))

2. conllu (`pip install conllu`), pandas (`pip install pandas`), numpy (`pip install numpy`)

# Code organization

## Training

### n-gram model

**bayes_model_hindi_train.py <max_sentences> <n> <save_name>** trains a **n**-gram model based on the first **max_sentences** sentences of the above-mentioned corpus and saves the saved json model as the specified **save_name**. By default, we use a 4-gram model on the first 5 million sentences. Sentences are simplified based on the scheme mentioned in the paper inside this. 

### Lossy-surprisal models

We don't need to learn the probability values of these models in the same way as an n-gram model. Rather, we can infer these on the fly during testing itself. The probabilities of only the n-gram model and corresponding noise distributions (**noise_distrs.py**) need to be specified and the rest, one can infer from the Bayesian network itself. 

## Testing

**bayes_model_hindi_test.py** finds the top predictions made by the n-gram model for each condition listed in the file **test_file.txt**. Similarly, **lossy_surp_model_test.py** finds the top predictions for the lossy-surprisal models for the conditions in the **test_file.txt**. One can change the various hyperparameters (which noise distribution to use, minimum probability to consider, how many top predictions etc.) inside these files.

**select_top.py** selects the top 50 predictions of various models. 

## Annotation

We have created a manually annotated database **verb_class_database.json** and listed the valid grammatical completions **valid_completions_per_condition.json**. Using this, one can annotate the predictions into various verb classes.

## Analysis

**analysis.py <model_name>** finds the percentage of grammatical completions per condition, distribution of error types per condition, and the KL-divergence and recall values per condition (and over all) of the predictions of the "**model_name**" model. It stores them into various csv files. 

Then, **plots.R** reads these csv files and plots various figures. 

# Note

Note that statistics on human predictions (from the completion study) are not provided in this code. 