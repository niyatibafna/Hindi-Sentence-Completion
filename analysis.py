import json
import pandas as pd
import numpy as np
import sys

def normalize (pw_r):
    s = 0
    for w, p in pw_r.items():
        s += p
    for w in pw_r:
        pw_r[w] /= s
    return pw_r

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

def find_etype (cond, vclass):
    if (vclass in ["N DT", "T"]): #CAUS too but it will not be an error
        # compatible = ["ne ko", "ne se", "ko ne", "se ne"]
        ne_ind = cond.split().index("ne")
        ko_ind = cond.split().index("ko")
        se_ind = cond.split().index("se")
        ne_ko = "N" + str(ne_ind+1) + "-" + "N" + str(ko_ind+1) if (ne_ind < ko_ind) else "N" + str(ko_ind+1) + "-" + "N" + str(ne_ind+1)
        ne_se = "N" + str(ne_ind+1) + "-" + "N" + str(se_ind+1) if (ne_ind < se_ind) else "N" + str(se_ind+1) + "-" + "N" + str(ne_ind+1)
        return ([ne_ko, ne_se])
    elif (vclass in ["N Pass", "Pass"]):
        # compatible = ["ko se", "se ko"]
        ko_ind = cond.split().index("ko")
        se_ind = cond.split().index("se")
        return [("N" + str(se_ind+1) + "-" + "N" + str(ko_ind+1) if (se_ind < ko_ind) else "N" + str(ko_ind+1) + "-" + "N" + str(se_ind+1))]
    else:
        return ["Random"]
    
# 
# 
# 
n_np = 3
if (sys.argv[1] == "4gram"):
    preds_file_name = "top_100predictions4gram_animate_verb_classes.json"
    pr = "prob"
elif (sys.argv[1] == "lc_rand"):
    preds_file_name = "top_100predictions_lossy_rand_erasure_subitem_10_125_verb_classes.json"
    pr = "probs"
elif (sys.argv[1] == "lc_pred"):
    preds_file_name = "top_100predictions_lossy_pred_bias2_subitem_10_125_verb_classes.json"
    pr = "probs"
elif (sys.argv[1] == "4gram_2np"):
    preds_file_name = "top_100np2_predictions4gram_animate_verb_classes.json"
    pr = "prob"
    n_np = 2
elif (sys.argv[1] == "lc_rand_2np"):
    preds_file_name = "top_100np2_predictions_lossy_rand_erasure_subitem_10_50_verb_classes.json"
    pr = "probs"
    n_np = 2
elif (sys.argv[1] == "lc_pred_2np"):
    preds_file_name = "top_100np2_predictions_lossy_pred_bias2_subitem_10_50_verb_classes.json"
    pr = "probs"
    n_np = 2
else:
    exit("Give model name")
print(preds_file_name)
# preds_file_name = "top_100predictions_lossy_pred_bias2_subitem_10_125_verb_classes.json"
# "top_100predictions4gram_animate_verb_classes.json"
# "top_100predictions_lossy_rand_erasure_subitem_10_125_verb_classes.json"
# "top_100predictions_lossy_pred_bias_subitem_10_125_verb_classes.json"
# "top_100predictions_lossy_pred_bias2_subitem_10_125_verb_classes.json"
if ("4gram" in preds_file_name):
    save_name = "4gram" if (n_np == 3) else "4gram_2np"
elif ("lossy_rand_erasure" in preds_file_name):
    save_name = "lossy_rand_erasure" if (n_np == 3) else "lossy_rand_erasure_2np"
elif ("lossy_pred_bias2" in preds_file_name):
    save_name = "lossy_pred_bias2" if (n_np == 3) else "lossy_pred_bias2_2np"
elif ("lossy_pred_bias" in preds_file_name):
    save_name = "lossy_pred_bias" if (n_np == 3) else "lossy_pred_bias_2np"
else:
    save_name = "idk"

grammatical_classes = {"ne ko": ["N DT", "CAUS", "T"], "ne se": ["N DT", "CAUS", "T"],
                        "ko ne": ["N DT", "CAUS", "T"], "se ne": ["N DT", "CAUS", "T"],
                        "ko se": ["N Pass", "Pass"], "se ko": ["N Pass", "Pass"], 
                        "ne ko se": ["CAUS", "DT"], "ne se ko": ["CAUS", "DT"], 
                        "ko ne se": ["CAUS", "DT"], "ko se ne": ["CAUS", "DT"],
                        "se ko ne": ["CAUS", "DT"], "se ne ko": ["CAUS", "DT"]}

predictions = json.load(open(preds_file_name, "r"))
new_predictions = {}
for cond in predictions:
    new_predictions[readable(cond)] = predictions[cond]
predictions = new_predictions

# Grammaticality
gram_probs = {}
for cond in predictions:
    gram_probs[cond] = {"Gram": 0, "Ungram": 0}

for cond in predictions:
    for pred in predictions[cond]:
        if ("verb_class" in predictions[cond][pred]):
            if (predictions[cond][pred]["verb_class"] in grammatical_classes[cond]):
                if ("probs" in predictions[cond][pred]):
                    gram_probs[cond]["Gram"] += sum(predictions[cond][pred]["probs"]) 
                else:
                    gram_probs[cond]["Gram"] += predictions[cond][pred]["prob"]
            else:
                if ("probs" in predictions[cond][pred]):
                    gram_probs[cond]["Ungram"] += sum(predictions[cond][pred]["probs"]) 
                else:
                    gram_probs[cond]["Ungram"] += predictions[cond][pred]["prob"]

for cond in gram_probs:
    gram_probs[cond] = normalize(gram_probs[cond])

gram_df = pd.DataFrame({"Condition": list(gram_probs.keys()), "Frequency": [gram_probs[cond]["Gram"] for cond in gram_probs]})
gram_df.to_csv('gram_' + save_name + ".csv", index=False)
# 

print(save_name)

if (n_np == 2):
    exit()

# Discussing error types

errors = ["N1-N2", "N2-N3", "N1-N3", "Random"]
error_distrs = {}

for cond in predictions:
    error_distrs[cond] = {}
    for etype in errors:
        error_distrs[cond][etype] = 0

for cond in predictions:
    for pred in predictions[cond]:
        if ("verb_class" in predictions[cond][pred]):
            if (predictions[cond][pred]["verb_class"] not in grammatical_classes[cond]):
                etypes = find_etype(cond, predictions[cond][pred]["verb_class"])
                if ("4gram" in preds_file_name):
                    if ("N2-N3" in etypes):
                        etypes = ["N2-N3"]
                for etype in etypes:
                    if ((etype != "N2-N3") and ("4gram" in preds_file_name)):
                        error_distrs[cond]["Random"] += 1
                    else:
                        error_distrs[cond][etype] += 1

for cond in error_distrs:
    error_distrs[cond] = normalize(error_distrs[cond])

error_df = {"Condition": [], "Error Type": [], "Frequency": []}
for cond in error_distrs:
    for etype in error_distrs[cond]:
        error_df["Condition"].append(cond)
        error_df["Error Type"].append(etype)
        error_df["Frequency"].append(error_distrs[cond][etype])

pd.DataFrame(error_df).to_csv('error_class_' + save_name + ".csv", index=False)
# 


# KL-Divergence

def kl_divergence(p, q):
    return np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))

human_class_preds = pd.read_csv("human_class_predictions.csv", index_col="Condition")

model_class_preds = {}
for cond in predictions:
    model_class_preds[cond] = {}
    for vclass in human_class_preds.columns:
        model_class_preds[cond][vclass] = 0
    for pred in predictions[cond]:
        if (("verb_class" in predictions[cond][pred]) and (predictions[cond][pred]["verb_class"] in human_class_preds.columns)):
            if ("probs" in predictions[cond][pred]):
                model_class_preds[cond][predictions[cond][pred]["verb_class"]] += sum(predictions[cond][pred]["probs"])  
            else:
                model_class_preds[cond][predictions[cond][pred]["verb_class"]] += predictions[cond][pred]["prob"]

for cond in model_class_preds:
    model_class_preds[cond] = normalize(model_class_preds[cond])

for cond in model_class_preds:
    print(cond)
    top_preds = [k for k, v in sorted(model_class_preds[cond].items(), 
                    key=lambda item: item[1], reverse=True)][:5]
    top_reals = [k for k, v in sorted(human_class_preds.loc[cond].items(), 
                    key=lambda item: item[1], reverse=True)][:5]
    for m_vc, h_vc in zip(top_preds, top_reals):
        print(m_vc, "|", h_vc)
    print("\n")

human_distr = []
model_distr = []
tp_global = 0.0
fn_global = 0.0
recall_df = {"Condition": [], "Recall": []}
divergence_df = {"Condition": [], "KLdiv_h_m": [], "KLdiv_m_h": []}
for cond in model_class_preds:
    human_distr_cond = []
    model_distr_cond = []
    for vclass in model_class_preds[cond]:
        human_distr_cond.append(human_class_preds.loc[cond][vclass])
        model_distr_cond.append(model_class_preds[cond][vclass])
        human_distr.append(human_class_preds.loc[cond][vclass])
        model_distr.append(model_class_preds[cond][vclass])
    human_distr_cond = np.array(human_distr_cond)
    model_distr_cond = np.array(model_distr_cond)
    divergence_df["Condition"].append(cond)
    divergence_df["KLdiv_h_m"].append(kl_divergence(human_distr_cond, model_distr_cond))
    divergence_df["KLdiv_m_h"].append(kl_divergence(model_distr_cond, human_distr_cond))
    recall_df["Condition"].append(cond)
    fn, tp = 0, 0
    for vclass in human_class_preds.columns:
        if (human_class_preds.loc[cond][vclass] != 0):
            if (model_class_preds[cond][vclass] == 0):
                fn += human_class_preds.loc[cond][vclass]
            else:
                tp += human_class_preds.loc[cond][vclass]
    tp_global += tp
    fn_global += fn
    recall_df["Recall"].append(tp/(fn+tp))

pd.DataFrame(divergence_df).to_csv("kl_div_" + save_name + ".csv", index=False)
pd.DataFrame(recall_df).to_csv("recall_" + save_name + ".csv", index=False)

print(kl_divergence(np.array(human_distr), np.array(model_distr)))
print(tp_global/(tp_global + fn_global))