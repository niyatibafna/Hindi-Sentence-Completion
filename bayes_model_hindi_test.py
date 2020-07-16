import json
import sys
from isc_tokenizer import Tokenizer
from isc_tagger import Tagger
from functools import reduce
import os

tk = Tokenizer(lang="hin")
pos_tagger = Tagger(lang="hin")
max_pred_length = 2
allowed_cases = ["ने", "को", "से"]
topk = 1000000
min_prob = 1e-5 # thresh

model_fname = "model_5000000_4gram_animacy_all.json"
n_gram = 4

model = json.load(open(model_fname, "r"))

pred_file_name = "np2_predictions4gram_animate_" + str(topk) + ".json"

if (os.path.exists(pred_file_name)):
    predictions = json.load(open(pred_file_name,"r"))
else: 
    predictions = {}

def reduce_by_one (context):
    return reduce(lambda x, y: (x+' '+y) if (x != '') else y, tk.tokenize(context)[1:], '')

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

def make_prediction (condition, context, last_prob=1.0, prediction="", n_preds=0, context_preds=[]):
    if ((n_preds >= max_pred_length) or (last_prob < min_prob)):
        predictions[condition][prediction[1:]] = {"contexts": context_preds, "prob": last_prob}
    elif ((context_len(context) <= (n_gram-1)) and (context in model)):
        for pred, prob in model[context].items():
            if ((not(pred.startswith("N"))) and (not(pred.startswith("A"))) and (pos_tagger.tag([pred])[0][1] == "SYM")):
                predictions[condition][(prediction + " " + pred)[1:]] = {"contexts": context_preds + [context_len(context)], "prob": prob * last_prob}
            else:
                make_prediction (condition, context + " " + pred, last_prob*prob, prediction + " " + pred, n_preds+1, context_preds + [context_len(context)])
    else:
        new_context = reduce_by_one(context)
        if (new_context == ""):
            predictions[condition][prediction[1:]] = {"contexts": context_preds, "prob": last_prob}
        else:
            make_prediction (condition, new_context, last_prob, prediction, n_preds, context_preds)

with open("test_file.txt", "r") as f:
    for text in f:
        test_condition = text[:-1]
        print(test_condition)
        predictions[test_condition] = {}
        make_prediction(test_condition, test_condition)
        top_preds = [k for k, v in sorted(predictions[test_condition].items(), key=lambda item: item[1]["prob"], reverse=True)][:topk]
        top_preds_condition = {}
        for pred in top_preds:
            top_preds_condition[pred] = predictions[test_condition][pred]
        predictions[test_condition] = top_preds_condition

json.dump(predictions, open(pred_file_name,"w+"), ensure_ascii=False)