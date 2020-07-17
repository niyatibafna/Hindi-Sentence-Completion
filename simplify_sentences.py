# %%
#accepts a file of raw_data, simplifies each sentences according to process mentioned in paper

from __future__ import unicode_literals
import parserz
import tokenizer
from parserz.isc_parser import Parser
from tokenizer import *
import tokenizer.isc_tokenizer
from tokenizer.isc_tokenizer import Tokenizer
from isc_tagger import Tagger

#positions of different args of the dependency parse 
index = 0
lex_item = 1
POS = 4
parent = 6
dep_rel = 7

# %%
#READING INPUT
input_file = open("raw_data.txt","r")
simplified_data = open("simplified_data.txt","w")
#Sentence tokenize data 
tk = Tokenizer(lang='hin', split_sen = True)
text_file = input_file.read()
text_data = tk.tokenize(text_file)
# print(text_data)


# %%
def extract_information(sentence):
    parser = Parser(lang='hin')
    tree = parser.parse(sentence)
    print('\n'.join(['\t'.join(node) for node in tree]))
    #Finding the (verb) root
    main_index = '0'
    heads = {'0'}
    for i in range(len(tree)):
        row = tree[i]
        if(row[parent]=='0' and row[dep_rel]=='main'):
            main_index = row[index]
    print(main_index)
    heads.add(main_index)
    #If we have a conjunct structure, there is a possibility of noun ellipsis. If we identify it, we simply truncate the sentence.
    if(tree[int(main_index)-1][POS]=='CC'):
        for i in range(len(tree)):
            row = tree[i]
            if(row[parent]==main_index and row[POS]=='VM'): #Find verb children - roots of S1 S2 subtrees
                print(row)
                nsubj = False
                for j in range(len(tree)):
                    if(tree[j][parent]==str(i+1) and tree[j][dep_rel]=='k1'): #Subject
                        nsubj = True
                if(nsubj):
                    heads.add(tree[i][index]) #Add verb to heads to pick up its dependents
                else:
                    if(i>int(main_index)):
                        heads.remove(main_index)
                        new_sentence=[]
                        for i in range(int(main_index)-1):
                            new_sentence.append(tree[i][lex_item])
                        if(tree[len(tree)-1][POS]=='SYM'):
                            new_sentence.append(tree[len(tree)-1][lex_item])
                        print(new_sentence)
                        return(extract_information(new_sentence)) #Recursion on truncated sentence


    #Finding the children of the root, as the heads of required phrases
    acceptable_tags = {'NN','NNP','QC', 'PRP','PSP','VM', 'VAUX', 'JJ','CC', 'SYM'} 
    #excluding NLoc, which is adverbial, like 'ghar mein' 
    #including JJ, because it may be pof
    #including QC, for नौ घर आए। etc. Technically we have an ellipsis
    acceptable_rel = {'k1','k2','pof', 'k2p','k3','k4','k5','lwg__psp', 'lwg__vaux', 'lwg__vaux_cont', 'ccof', 'main', 'rsym'}
    #These might not pick up compound verb formations: adding 'pof' to pick up noun parts of CV
    for i in range(len(tree)):
        row = tree[i]
        if(row[parent]==main_index and row[POS] in acceptable_tags and row[dep_rel] in acceptable_rel):
            heads.add(row[index])
    print("Heads: ")
    print(heads)
    #Collecting correct dependents, like noun and pp arguments, of heads, and building a set of correct indices
    indices = set()
    for i in range(len(tree)):
        row = tree[i]
        if(row[parent] in heads and row[POS] in acceptable_tags and row[dep_rel] in acceptable_rel):
            indices.add(i+1) #note that these are tree list indices: i = row index - 1
    print("Collected indices: ")
    print(indices)
    #OPTIONAL: Going one further level down for safety
    for i in range(len(tree)):
        row = tree[i]
        if((int)(row[parent]) in indices and row[POS] in acceptable_tags and row[dep_rel] in acceptable_rel):
            indices.add(i+1)
    #Building simplified sentence
    simplified_sent = ""
    for i in range(len(tree)):
        if((i+1) in indices):
            simplified_sent += tree[i][lex_item] + " "
    if(tree[len(tree)-1][POS]=='SYM'):
        simplified_sent += (tree[len(tree)-1][lex_item]) + "\n"

    print("Simplified sentences: ")
    print(simplified_sent)
    return(simplified_sent)


# %%

#For each sentence
for sentence in text_data:
    simplified_sent = extract_information(sentence)
    simplified_data.write("Original:\n")
    simplified_data.write(" ".join(sentence))
    simplified_data.write("\n"+"Simplified:\n"+ simplified_sent+"\n")
    

# %%
