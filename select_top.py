import pandas as pd 
import csv
import json
import numpy as np
from isc_tokenizer import Tokenizer 
from isc_tagger import Tagger

tk = Tokenizer(lang="hin")
pos_tagger = Tagger(lang="hin")
allowed_cases = ["ने", "को", "से"]

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

def readable(cond):
    new_cond = ""
    case_markers = ["ने", "को", "से"]
    for s in cond.split(" "):
        if (s ==  "ने"):
            new_cond += "ne "
        elif (s == "को"):
            new_cond += "ko "
        elif (s == "से"):
            new_cond += "se "
    return new_cond[:-1]

topk = 1000000
top_topk = 20

model_lex_preds = json.load(open("predictions4gram_animate_" + str(topk) + ".json", "r"))
conditions = ["A ने A को A से", "A ने A से A को", "A को A ने A से", "A को A से A ने", 
              "A से A को A ने", "A से A ने A को"]

top_topk = 100
top_topk_predictions = {}
for condition in conditions:
    top_preds = [k for k, v in sorted(model_lex_preds[condition].items(), key=lambda item: item[1]["prob"], reverse=True)][:top_topk]
    top_topk_predictions[condition] = {k:model_lex_preds[condition][k] for k in top_preds}
    top_topk_predictions[condition] = sorted(top_topk_predictions[condition].items(), key=lambda x: x[1]['prob'], reverse=True)

json.dump(top_topk_predictions, open("top_" + str(top_topk) + "predictions4gram_animate.json", "w+"), ensure_ascii=False)

model_lex_preds = json.load(open("predictions_lossy_pred_bias2_subitem_10_125.json", "r"))
conditions = ["A ने A को A से", "A ने A से A को", "A को A ने A से", "A को A से A ने", 
              "A से A को A ने", "A से A ने A को"]

reduced_model_lex_preds = {}
for cond in conditions:
    reduced_model_lex_preds[cond] = {}
    for pred in model_lex_preds[cond]:
        pred_tokens = context_tokens(pred)
        if ((len(pred_tokens) == 3) or ((not(pred_tokens[-1].startswith("N"))) and (not(pred_tokens[-1].startswith("A"))) and (pos_tagger.tag([pred_tokens[-1]])[0][1] == "SYM"))):
            reduced_model_lex_preds[cond][pred] = model_lex_preds[cond][pred]

model_lex_preds = reduced_model_lex_preds

for cond in conditions:
    print(cond, len(model_lex_preds[cond]))

top_topk = 100

top_topk_predictions = {}
for condition in conditions:
    top_preds = [k for k, v in sorted(model_lex_preds[condition].items(), key=lambda item: np.sum(item[1]['probs'])/10, reverse=True)][:top_topk]
    top_topk_predictions[condition] = {k:model_lex_preds[condition][k] for k in top_preds}
    top_topk_predictions[condition] = sorted(top_topk_predictions[condition].items(), key=lambda x: np.sum(x[1]['probs']), reverse=True)

json.dump(top_topk_predictions, open("top_" + str(top_topk) + "predictions_lossy_pred_bias2_subitem_10_125.json", "w+"), ensure_ascii=False)

# 
# 2-NP
# 
topk=1000000
model_lex_preds = json.load(open("np2_predictions4gram_animate_" + str(topk) + ".json", "r"))
conditions = ["A ने A को", "A ने A से", "A को A ने", "A को A से", "A से A को", "A से A ने"]

top_topk = 100
top_topk_predictions = {}
for condition in conditions:
    top_preds = [k for k, v in sorted(model_lex_preds[condition].items(), key=lambda item: item[1]["prob"], reverse=True)][:top_topk]
    top_topk_predictions[condition] = {k:model_lex_preds[condition][k] for k in top_preds}
    top_topk_predictions[condition] = sorted(top_topk_predictions[condition].items(), key=lambda x: x[1]['prob'], reverse=True)

json.dump(top_topk_predictions, open("top_" + str(top_topk) + "np2_predictions4gram_animate.json", "w+"), ensure_ascii=False)

model_lex_preds = json.load(open("np2_predictions_lossy_pred_bias2_subitem_10_50.json", "r"))
conditions = ["A ने A को", "A ने A से", "A को A ने", "A को A से", "A से A को", "A से A ने"]

reduced_model_lex_preds = {}
for cond in conditions:
    reduced_model_lex_preds[cond] = {}
    for pred in model_lex_preds[cond]:
        pred_tokens = context_tokens(pred)
        if ((len(pred_tokens) == 2) or ((not(pred_tokens[-1].startswith("N"))) and (not(pred_tokens[-1].startswith("A"))) and (pos_tagger.tag([pred_tokens[-1]])[0][1] == "SYM"))):
            reduced_model_lex_preds[cond][pred] = model_lex_preds[cond][pred]

model_lex_preds = reduced_model_lex_preds

for cond in conditions:
    print(cond, len(model_lex_preds[cond]))

top_topk = 100

top_topk_predictions = {}
for condition in conditions:
    top_preds = [k for k, v in sorted(model_lex_preds[condition].items(), key=lambda item: np.sum(item[1]['probs'])/10, reverse=True)][:top_topk]
    top_topk_predictions[condition] = {k:model_lex_preds[condition][k] for k in top_preds}
    top_topk_predictions[condition] = sorted(top_topk_predictions[condition].items(), key=lambda x: np.sum(x[1]['probs']), reverse=True)

json.dump(top_topk_predictions, open("top_" + str(top_topk) + "np2_predictions_lossy_pred_bias2_subitem_10_50.json", "w+"), ensure_ascii=False)
