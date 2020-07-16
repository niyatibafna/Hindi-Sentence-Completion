import json
import numpy as np
from noise_distrs import *
from isc_tokenizer import Tokenizer
from isc_tagger import Tagger
from functools import reduce
from collections import defaultdict
import pymongo
import sys
import os

tk = Tokenizer(lang="hin")
pos_tagger = Tagger(lang="hin")
max_pred_length = 4
topk = 1000000
min_prob = 1e-5 # thresh
max_iters = 5
topk_prob = 50 #lambda x: 5**(max_pred_length - x) # somewhat simulates the no. of items

n_subjects = 10

context_size = 3

l_size = 2

# model_fname = sys.argv[1]
model_fname = "model_5000000_4gram_animacy_all.json"
model = json.load(open(model_fname, "r"))

client = pymongo.MongoClient()
db = client.model_db
sentence_collection = db.sentence_collection
n = len(list(sentence_collection.find()))
print(n)
if (n == 0):
    for key in model:
        preds = model[key].copy()
        preds = {k:v for k, v in preds.items() if (('.' not in k) and (not(k.startswith('$'))))}
        preds['_id'] = key
        try:
            insert_results = sentence_collection.insert_one(preds)
        except:
            pass

print("Loaded models")

# Noise = RandErasure(d=0.1)
# Noise = RecencyBiasErasure(nd=1, decay_factor=1-d)
# Noise = PredictableBiasErasure(model, context_size=context_size, decay_factor=1-d, min_prob=min_prob)
Noise = PredictableReductionBias(model, context_size=context_size, min_prob=min_prob, ce_penalty=0)
# changed to predictable but at least the last token is not lost.
allowed_cases = ["ने", "को", "से"]
unique_cases = ['A ने', 'A को', 'A से']

pred_file_name = "predictions_lossy_pred_bias2_subitem_" + str(n_subjects) + "_" + str(topk_prob) + ".json"

if (os.path.exists(pred_file_name)):
    predictions = json.load(open(pred_file_name,"r"))
else: 
    predictions = {}

def isSubSequence(string1, string2): 
    m = len(string1) 
    n = len(string2) 
    if m == 0:    
        return True
    if n == 0:    
        return False
    if string1[m-1] == string2[n-1]: 
        return isSubSequence(string1[:m-1], string2[:n-1])
    return isSubSequence(string1[:m], string2[:n-1])

def context_tokens (context):
    tokens = tk.tokenize(context)
    new_tokens = []
    i = 0
    while (i < len(tokens)):
        if (((tokens[i] == "N") or (tokens[i] == "A")) and (i < (len(tokens) - 1)) and (tokens[i+1] in allowed_cases)):
            new_tokens.append(tokens[i] + " " + tokens[i+1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def case_exchange_tokens (tokens):
    all_contexts = [tokens]
    for i, token in enumerate(tokens):
        if (token in unique_cases):
            for case_token in unique_cases:
                if (case_token != token):
                    r_ce = tokens.copy()
                    r_ce[i] = case_token
                    all_contexts.append(r_ce)
    return all_contexts

def context_len(context):
    tokens = tk.tokenize(context)
    clen = 0
    i = 0
    while (i < len(tokens)):
        clen += 1
        if (((tokens[i] == "N") or (tokens[i] == "A")) and (i < (len(tokens) - 1)) and (tokens[i+1] in allowed_cases)):
            i += 2
        else:
            i += 1
    return clen

def prob_context (real_context):
    real_tokens = context_tokens(real_context)
    p = 1
    for i in range(len(real_tokens)):
        try:
            p *= model[' '.join(real_tokens[i-context_size-1:i])][real_tokens[i]]
        except:
            p = 0
    return p

def normalize (pw_r):
    s = 0
    for w, p in pw_r.items():
        s += p
    for w in pw_r:
        pw_r[w] /= s
    return pw_r

def make_prediction (condition, context, last_prob=1.0, prediction="", n_preds=0, context_preds=[], store_last=False):
    global predictions
    if (last_prob < min_prob):
        return
    if (n_preds >= max_pred_length):
        if (prediction[1:] not in predictions[condition]):
            predictions[condition][prediction[1:]] = {"contexts": [context_preds], "probs": [last_prob]}
        else:
            predictions[condition][prediction[1:]] = {"contexts": predictions[condition][prediction[1:]]["contexts"] + [context_preds], 
                                                    "probs": predictions[condition][prediction[1:]]["probs"] + [last_prob]}
    elif ((context_len(context) <= l_size) and (context in model)):
        # L
        for pred, prob in sorted(model[context].items(), key=lambda item:item[1], reverse=True)[:topk_prob]:#[:topk_prob(n_preds+1)]:
            if ((not(pred.startswith("N"))) and (not(pred.startswith("A"))) and (pos_tagger.tag([pred])[0][1] == "SYM")):
                if ((last_prob*prob) < min_prob):
                    return
                new_prediction = (prediction + " " + pred)[1:]
                if (new_prediction not in predictions[condition]):
                    predictions[condition][new_prediction] = {"contexts": [context_preds + [context]], 
                                                            "probs": [prob * last_prob]}
                else:
                    predictions[condition][new_prediction] = {"contexts": predictions[condition][new_prediction]["contexts"] + [context_preds + [context]], 
                                                            "probs": predictions[condition][new_prediction]["probs"] + [prob * last_prob]}                    
            else:
                make_prediction (condition, context + " " + pred, last_prob*prob, prediction + " " + pred, n_preds+1, context_preds + [context], store_last=True)
    else:
        # L-M
        print(context)
        fit_sentences = []
        n_iters = 0
        while ((n_iters < max_iters) and (len(fit_sentences) == 0)):
            try:
                reduced_context_tokens = Noise.add_noise(context_tokens(context), store_last=store_last)
            except:
                continue
            reduced_context = ' '.join(reduced_context_tokens)
            if ((len(reduced_context_tokens) == 0)): #or ((len(context_preds) >= 1) and (reduced_context == context_preds[-1]))):
                continue
            pw_r = defaultdict(lambda: 0)
            regex_expr = '.*'.join(reduced_context_tokens)
            fit_sentences = list(sentence_collection.find({"_id": { "$regex": regex_expr }}))
            n_iters += 1
        print(context, ' reduced to ', reduced_context)
        for sentence in fit_sentences:
            real_context = sentence['_id']
            pr_c = Noise.pr_c(reduced_context_tokens, context_tokens(real_context), store_last=store_last)
            pc = prob_context(real_context)
            for w, pw_c in sentence.items():
                if (w != '_id'):
                    p = pr_c * pc * pw_c
                    if (p != 0):
                        pw_r[w] += p 
        pw_r = normalize(pw_r)
        if (len(pw_r.items()) == 0):
            return
        for pred, prob in sorted(pw_r.items(), key=lambda item:item[1], reverse=True)[:topk_prob]:#[:topk_prob(n_preds+1)]:
            if ((pred != '') and (not(pred.startswith("N"))) and (not(pred.startswith("A"))) and (pos_tagger.tag([pred])[0][1] == "SYM")):
                if ((last_prob*prob) < min_prob):
                    return
                new_prediction = (prediction + " " + pred)[1:]
                if (new_prediction not in predictions[condition]):
                    predictions[condition][new_prediction] = {"contexts": [context_preds + [reduced_context]], 
                                                            "probs": [prob * last_prob]}
                else:
                    predictions[condition][new_prediction] = {"contexts": predictions[condition][new_prediction]["contexts"] + [context_preds + [reduced_context]], 
                                                            "probs": predictions[condition][new_prediction]["probs"] + [prob * last_prob]}                    
            else:
                make_prediction (condition, reduced_context + " " + pred, last_prob*prob, prediction + " " + pred, 
                                n_preds+1, context_preds + [reduced_context], store_last=True)


with open("test_file.txt", "r") as f:
    for text in f:
        test_condition = text[:-1]
        print(test_condition)
        predictions[test_condition] = {}
        for subj in range(n_subjects):
            print(subj)
            make_prediction(test_condition, test_condition)
        print(len(predictions[test_condition].keys()))
        # top_preds = [k for k, v in sorted(predictions[test_condition].items(), key=lambda item: item[1]["prob"], reverse=True)][:topk]
        # top_preds_condition = {}
        # for pred in top_preds:
        #     top_preds_condition[pred] = predictions[test_condition][pred]
        # predictions[test_condition] = top_preds_condition
        json.dump(predictions, open(pred_file_name,"w+"), ensure_ascii=False)

