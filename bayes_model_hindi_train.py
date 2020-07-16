from collections import defaultdict
from isc_parser import Parser
from isc_tagger import Tagger
from isc_tokenizer import Tokenizer
import sys
import json
from functools import reduce
import os
from io import open
import pandas as pd
from conllu import parse as conllu_parse
from csv import DictWriter

model = defaultdict(lambda: defaultdict(lambda: 0))

tk = Tokenizer(lang='hin', split_sen=True)
tagger = Tagger(lang='hin')
parser = Parser(lang='hin')

allowed_cases = ["ने", "को", "से"]
min_sent_length = 5

pred_contexts = ["A ने A को", "A ने A से", "A को A ने", "A को A से", "A से A को", 
                "A से A ने", "A ने A को A से", "A ने A से A को", "A को A ने A से", 
                "A को A से A ने", "A से A को A ने", "A से A ने A को"]
                
max_sentences = int(sys.argv[1]) if (len(sys.argv) > 1) else 5000000
n_gram = int(sys.argv[2]) if (len(sys.argv) > 2) else 3
save_name = sys.argv[3] if (len(sys.argv) > 3) else "model_animate_" + str(max_sentences) + "_" + str(n_gram) + "gram.json"
# offset = int(sys.argv[3]) if (len(sys.argv) > 3) else 0
print(save_name) #, offset)

animate_nouns = []
with open ("animate_nouns.txt", "r") as f:
    for text in f:
        animate_nouns.append(text[:-1])

animate_pronouns = []
animates = animate_nouns + animate_pronouns

def parse_sentence (sent):
    tree = parser.parse(sent)
    parsed_tree = conllu_parse('\n'.join(['\t'.join(node) for node in tree]))
    return parsed_tree[0]
    

def simplify_sentence(tokened_sentence, deps_depth=2):
    sent_tokens = parse_sentence(tokened_sentence)
    sent_tree = sent_tokens.to_tree()
    main_index = sent_tree.token['id']
    allowed_indices = []
    allowed_indices.append(main_index)
    #If we have a conjunct structure, there is a possibility of noun ellipsis. If we identify it, we simply truncate the sentence.
    if(sent_tree.token['upostag']=='CC'):
        for main_child in sent_tree.children:
            mc_token = main_child.token
            if(mc_token['upostag']=='VM'): #Find verb children - roots of S1 S2 subtrees
                nsubj = False
                for mcc_sent_tree in main_child.children:
                    if (mcc_sent_tree.token['deprel'] == 'k1'):
                        nsubj = True
                        break
                if(nsubj):
                    allowed_indices.append(mc_token['id']) #Add verb to heads to pick up its dependents
                else:
                    if(mc_token['id'] > main_index):
                        new_sentence=[]
                        for i in range(int(main_index)-1):
                            new_sentence.append(sent_tokens[i]['lemma'])
                        if(sent_tokens[len(sent_tree)-1]['upostag']=='SYM'):
                            new_sentence.append(sent_tokens[-1]['lemma'])
                        return(simplify_sentence(new_sentence)) #Recursion on truncated sentence
    #Finding the children of the root, as the heads of required phrases
    #excluding NLoc, which is adverbial, like 'ghar mein' 
    #including JJ, because it may be pof
    #including QC, for नौ घर आए। etc. Technically we have an ellipsis
    acceptable_tags = ['NN','NNP','QC', 'PRP','PSP','VM', 'VAUX', 'JJ','CC', 'SYM']
    #These might not pick up compound verb formations: adding 'pof' to pick up noun parts of CV
    acceptable_rel = ['k1','k2','pof', 'k2p','k3','k4','k5','lwg__psp', 'lwg__vaux', 
                        'lwg__vaux_cont', 'ccof', 'main', 'rsym']
    curr_level = sent_tree.children
    for i in range(deps_depth):
        new_level = []
        for curr_tree in curr_level:
            if ((curr_tree.token['upostag'] in acceptable_tags) and (curr_tree.token['deprel'] in acceptable_rel)):
                allowed_indices.append(curr_tree.token['id'])
                new_level += curr_tree.children
        curr_level = new_level
    #Building simplified sentence
    #simplified_sent_lemmas = []
    #simplified_sent_pos = []
    simplified_sent_tokens = []
    for token in sent_tokens:
        if(token['id'] in allowed_indices):
            simplified_sent_tokens.append(token)
    if ((simplified_sent_tokens[-1]["lemma"] != sent_tokens[-1]["lemma"]) and (sent_tokens[-1]['upostag']=='SYM')):
        simplified_sent_tokens.append(sent_tokens[-1])
    #else:
        #simplified_sent = simplified_sent[:-1]
    return(simplified_sent_tokens)

def spl_ngram_tokens (parsed_tokens, n, pad_right=False, pad_left=False, noun_case_merge=False, allowed_cases=None, tok_abstract=False):
    tokens = [ptoken['lemma'] for ptoken in parsed_tokens]
    #if ((allowed_cases is not None) or (noun_case_merge) or (tok_abstract)):
    #    parsed_tokens = parse_sentence(tokens)
    if (allowed_cases is not None):
        new_tokens = []
        disallowed_ids = []
        for ptoken in parsed_tokens:
            if ((ptoken['deprel'] == 'lwg__psp') and (ptoken['lemma'] not in allowed_cases)):
                disallowed_ids.append(ptoken['head'])
                disallowed_ids += [tok['id'] for tok in parsed_tokens if (tok['head'] == ptoken['head'])]
        for ptoken in parsed_tokens:
            if (ptoken['id'] not in disallowed_ids):
                new_tokens.append(ptoken['lemma'])
        tokens = new_tokens
    if (noun_case_merge or tok_abstract):
        new_tokens = []
        i = 0
        while (i < (len(parsed_tokens)-1)):
            if (noun_case_merge and (parsed_tokens[i]["upostag"] in ["NN", "NNP", "PRP"]) and (parsed_tokens[i+1]["deprel"] == "lwg__psp")):
                # if (tok_abstract and (parsed_tokens[i]["upostag"] == "NNP")):
                #     new_tokens.append("NNP" + " " + parsed_tokens[i+1]['lemma'])
                # else:
                #     new_tokens.append(parsed_tokens[i]['lemma'] + " " + parsed_tokens[i+1]['lemma'])
                if (tok_abstract):
                    if ((parsed_tokens[i]["lemma"] in animates) or (parsed_tokens[i]["upostag"] == "PRP")):
                        new_tokens.append("A " + parsed_tokens[i+1]['lemma'])
                    else:
                        new_tokens.append("N " + parsed_tokens[i+1]['lemma'])
                else:
                    new_tokens.append(parsed_tokens[i]['lemma'] + " " + parsed_tokens[i+1]['lemma'])
                i += 2
            elif (parsed_tokens[i]['upostag'] in ["NN", "NNP", "PRP"]):
                if (tok_abstract):
                    if ((parsed_tokens[i]["lemma"] in animates) or (parsed_tokens[i]["upostag"] == "PRP")):
                        new_tokens.append("A")
                    else:
                        new_tokens.append("N")
                else:
                    new_tokens.append(parsed_tokens[i]['lemma'])
                i += 1
            else:
                new_tokens.append(parsed_tokens[i]['lemma'])
                i += 1
        new_tokens.append(parsed_tokens[i]["lemma"])
        tokens = new_tokens
    if (pad_right):
        tokens = tokens + ([''] * (n - 1))
    if (pad_left):
        tokens = ([''] * (n - 1))+ tokens
    context_unit_tokens = []
    for k in range(2, n+1):
        for i in range(len(tokens)-k+1):
            context_unit_tokens.append({"context": reduce(lambda x, y: (x+' '+y) if (x != '') else y, tokens[i:i+k-1], ''), 
                                "unit": tokens[i+k-1]})
    return context_unit_tokens

with open("../../../HDMI_Hindi/hindi-data/IITB/monolingual.hi", "r", encoding= "utf-8") as f, open("log_file.csv", "w+", encoding="utf-8") as logf:
    count_sentences = 1
    n_lines = 0
    log_writer = DictWriter(logf, fieldnames=["real_sent", "simp_sent"])
    log_writer.writeheader()
    for text in f :
        if (count_sentences % 1000 == 0):
            print(count_sentences)
        if (count_sentences >= max_sentences):
            break
        else:
            try:
                token_sentences = tk.tokenize(text)
                # sentence = tk.tokenize(text)[0]
                for sentence in token_sentences:
                    log_df = {} 
                    if (len(sentence) >= min_sent_length):
                        # model_tokens = ngramify(sentence, n_gram, pad_left=True)
                        # model_tokens = noun_verbal_tokens(sentence, n_gram)
                        simp_sent_tokens = simplify_sentence(sentence)
                        # print(simp_sent_tokens)
                        model_tokens = spl_ngram_tokens(simp_sent_tokens, 
                                                        n_gram, pad_left=True, noun_case_merge=True, 
                                                        allowed_cases=allowed_cases, tok_abstract=True)
                        # print(model_tokens)
                        # print("\n")
                        for token in model_tokens:
                            # context = reduce(lambda x, y: (x+' '+y) if (x != '') else y, token[:-1], '')
                            if (token["context"] in pred_contexts):
                                log_df["real_sent"] = ' '.join(sentence)
                                log_df["simp_sent"] = reduce(lambda x, y: (x + " " + y) if (x != '') else y, [token['lemma'] for token in simp_sent_tokens], '')
                            model[token["context"]][token["unit"]] += 1
                        if (len(log_df.keys()) > 0):
                            log_writer.writerow(log_df)
                            log_df = {}
                        count_sentences += 1
            except:
                pass

for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

#print(model)

with open(save_name, "w+", encoding="utf-8") as f:
    f.write(json.dumps(model, ensure_ascii=False))
