import numpy as np
import itertools
from collections import defaultdict

# c is a list of tokens 
# r is a list of tokens

unique_cases = ['A ने', 'A को', 'A से']

def normalize (dist):
    s = 0
    if (type(dist) == dict):
        for w, p in dist.items():
            s += p
        for w in dist:
            dist[w] /= s
    else:
        s = sum(dist)
        dist = [p/s for p in dist]
    return dist

def count(X, Y):
	m, n = len(X), len(Y)
	# T[i][j] stores number of of times the pattern Y[0..j)
	# appears in given X[0..i) as a subsequence
	T = np.zeros(shape=(m+1, n+1))
	# if pattern Y is empty, we have found subsequence
	for i in range(m + 1):
		T[i][0] = 1
	# If current character of both and pattern matches,
	# 1. exclude current character in both and pattern
	# 2. exclude only current character in the String
	# else if current character of and pattern do not match,
	# exclude current character in the String
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			T[i][j] = (T[i - 1][j - 1] if (X[i - 1] == Y[j - 1]) else 0) + T[i - 1][j]
	# return last entry in lookup table
	return T[m][n]

class RandErasure:
    def __init__(self, d=0.01):
        self.d = d

    def pr_c (self, r, c, store_last=False):
        m, n = len(c), len(r)
        return (count(c, r) * (self.d ** (m - n)) * ((1 - self.d) ** (n-(1 if store_last else 0))))

    def add_noise (self, c, store_last=False):
        r = []
        last_word = c[-1]
        c = c[:-1] if (store_last) else c
        for w in c:
            if (np.random.uniform(0, 1) > self.d):
                r.append(w)
        return ((r + [last_word]) if (store_last) else r)

class RecencyBiasErasure:
    def __init__(self, nd=0.99, decay_factor=0.1):
        self.nd = nd
        self.decay_factor = decay_factor

    def __sum_prob(self, X, Y):
        m, n = len(X), len(Y)
        T = np.zeros(shape=(m+1, n+1))
        T[0, 0] = 1
        prob_nd_until_i = self.nd * self.decay_factor**(m-1)
        prob_nd_i = self.nd * self.decay_factor**(m-1)
        for i in range(1, m + 1):
            T[i][0] = 1 * (1 - prob_nd_until_i)
            prob_nd_i /= self.decay_factor
            prob_nd_until_i *= prob_nd_i
        prob_nd_i = self.nd * self.decay_factor**(m-1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if (X[i - 1] == Y[j - 1]):
                    T[i,j] = T[i-1][j-1]*prob_nd_i + T[i-1,j]*(1 - prob_nd_i)
                else:
                    T[i,j] = T[i-1,j]*(1 - prob_nd_i)
            prob_nd_i /= self.decay_factor
        return T[m][n]

    def pr_c (self, r, c):
        return (self.__sum_prob(c, r))

    def add_noise (self, c):
        r = []
        decay_i = self.decay_factor ** (len(c) - 1)
        for w in c:
            if (np.random.uniform(0, 1) <= self.nd*decay_i):
                r.append(w)
            decay_i /= self.decay_factor
        return (r)

class PredictableBiasErasure:
    def __init__(self, l_model, context_size=3, decay_factor=1, min_prob=1e-10):
        self.l_model = l_model
        self.context_size = context_size
        self.decay_factor = decay_factor
        self.min_prob = min_prob

    def __find_prob_l(self, X, Y):
        m, n = len(X), len(Y)
        T = np.zeros(shape=(m+1, n+1))
        T[0, 0] = 1
        decay_i = self.decay_factor ** (m-1)
        try:
            prob_nd_i = self.l_model[''][X[0]] * decay_i
        except:
            prob_nd_i = self.min_prob
        prob_nd_until_i = prob_nd_i
        for i in range(1, m + 1):
            T[i][0] = 1 * (1 - prob_nd_until_i)
            decay_i /= self.decay_factor
            try:
                prob_nd_i = self.l_model[' '.join(X[i-1+self.context_size:i-1])][X[i-1]]/decay_i
            except:
                prob_nd_i = self.min_prob
            prob_nd_until_i *= prob_nd_i
        decay_i = self.decay_factor ** (m-1)
        for i in range(1, m + 1):
            try:
                prob_nd_i = self.l_model[' '.join(X[i-1+self.context_size:i-1])][X[i-1]]/decay_i
            except:
                prob_nd_i = self.min_prob
            for j in range(1, n + 1):
                if (X[i - 1] == Y[j - 1]):
                    T[i,j] = T[i-1][j-1]*prob_nd_i + T[i-1,j]*(1 - prob_nd_i)
                else:
                    T[i,j] = T[i-1,j]*(1 - prob_nd_i)
            decay_i /= self.decay_factor
        return T[m][n]

    def pr_c (self, r, c):
        return (self.__find_prob_l(c, r))

    def add_noise (self, c):
        r = []
        decay_i = self.decay_factor ** (len(c) - 1)
        prob_nd_i_dict = {}
        for i, w in enumerate(c):
            try:
                prob_nd_i_dict[i] = self.l_model[' '.join(c[(i+self.context_size-1):i])][w] * decay_i
            except:
                prob_nd_i_dict[i] = self.min_prob
            decay_i /= self.decay_factor
        prob_nd_i_dict = normalize(prob_nd_i_dict)
        # Rather than being too much random noise: I believe it's better to use the topk
        # But the problem comes when no such case is actually available in our limited window model :(
        # top_i = [i for i, p in sorted(prob_nd_i_dict.items(), key=lambda item: item[1], reverse=True)][:(self.context_size-1)]
        # r = [w for i, w in enumerate(c) if (i in top_i)] + r
        for i, w in enumerate(c):
            if (np.random.uniform(0, 1) <= prob_nd_i_dict[i]):
                r.append(w) 
        # # Last one has to be used !?!?!
        # r.append(c[-1]) 
        return (r)


class PredictableReductionBias:
    def __init__(self, l_model, topk=3, context_size=3, min_prob=1e-10, ce_penalty=0.1, d_penalty=1, prob_red_csize=0.2):
        self.l_model = l_model
        self.context_size = context_size
        self.min_prob = min_prob
        self.ce_penalty = ce_penalty
        self.d_penalty = d_penalty
        self.total_prob_csize = {}
        self.topk = topk
        self.prob_red_csize = prob_red_csize
        for i in range(1, 10):
            self.total_prob_csize[i] = max(self.min_prob, self.__find_prob_csize('', i))

    def prob_context (self, tokens):
        p = 1
        for i in range(len(tokens)):
            try:
                p *= self.l_model[' '.join(tokens[i-self.context_size-1:i])][tokens[i]]
            except:
                p = self.min_prob
        return p

    def pr_c (self, r, c, store_last=False):
        if (store_last):
            if (r[-1] == c[-1]):
                return (count(c, r) * self.prob_context(r) * (self.prob_red_csize ** (len(c) - len(r))))
                # if (count(c, r) == 0):
                #     return (self.ce_penalty * self.prob_context(r))
                # else:
                #     return (count(c, r) * self.prob_context(r))
            else:
                return 0
        else:
            return (count(c, r) * self.prob_context(r) * (self.prob_red_csize ** (len(c) - len(r))))
            # if (count(c, r) == 0):
            #     return (self.ce_penalty * self.prob_context(r))
            # else:
            #     return (count(c, r) * self.prob_context(r))

    def __find_prob_csize (self, context, csize):
        if (csize == 0):
            return 1
        else:
            p = 0
            if (context in self.l_model):
                for token in self.l_model[context]:
                    if ((token != "") and (token in unique_cases)):
                        new_context = (context + " " + token) if (context != "") else token
                        p += self.l_model[context][token]*self.__find_prob_csize(new_context, csize-1)
            return p

    def __find_possible_rs (self, s):
        combs = []
        for l in range(1, len(s)+1):
            combs += [list(x) for x in list(itertools.combinations(s, l))]
        return combs

    def add_noise (self, c, store_last=False):
        if (store_last):
            chosen_r = [c[-1]]
            c = c[:-1]
        else:
            chosen_r = []
        csize = len(c) + 1
        prob_rs_c = []
        all_rs_c = []
        d_pen = 1
        while(csize > 2):
            csize -= 1
            csize_rs = [list(x) for x in list(itertools.combinations(c, csize))]
            # all_rs_c += csize_rs
            for r in csize_rs:
                r_prob = (self.prob_context(r) * d_pen)/self.total_prob_csize[csize]
                # rand = np.random.uniform(0, 1)
                # if (rand < r_prob):
                #     # print (rand, r_prob) #(r_prob/self.total_prob_csize[csize]))
                #     return (r + chosen_r)
                all_rs_c.append(r)
                prob_rs_c.append(r_prob)
            if (np.random.uniform(0, 1) > self.prob_red_csize):
                break
                # Only one case exchange allowed
                # if (csize == len(c)):
                #     for i, token in enumerate(r):
                #         if (token in unique_cases):
                #             for case_token in unique_cases:
                #                 if (case_token != token):
                #                     r_ce = r.copy()
                #                     r_ce[i] = case_token
                #                     r_prob_ce = self.prob_context(r_ce)*self.ce_penalty
                #                     rand = np.random.uniform(0, 1)
                #                     if (rand < r_prob_ce/self.total_prob_csize[csize]):
                #                         # print(rand, (r_prob_ce/self.total_prob_csize[csize]))
                #                         return (r_ce + chosen_r)
                #                     # all_rs_c.append(r_ce)
                #                     # prob_rs_c.append(r_prob_ce)
                # d_pen *= csize/(len(c))
                # d_pen *= self.d_penalty
        if (len(all_rs_c) == 1):
            return (all_rs_c[0] + chosen_r)
        else:
            inds = sorted(range(len(prob_rs_c)), key=lambda x: prob_rs_c[x], reverse=True)[:self.topk]
            all_rs_c = [all_rs_c[i] for i in inds]
            prob_rs_c = [prob_rs_c[i] for i in inds]
            prob_rs_c = normalize(prob_rs_c)
            print(list(zip(all_rs_c, prob_rs_c)))
            return (all_rs_c[np.random.choice(np.arange(self.topk), p=prob_rs_c)] + chosen_r)

# from noise_distrs import *
# import json
# from isc_tokenizer import Tokenizer
# model_fname = "model_5000000_4gram_animacy_all.json"
# model = json.load(open(model_fname, "r"))
# tk = Tokenizer(lang="hin")
# d = 0.1
# context_size = 3
# allowed_cases = ["ने", "को", "से"]
# Noise = PredictableReductionBias(model, context_size=context_size, topk=4)
# def context_tokens (context):
#     tokens = tk.tokenize(context)
#     new_tokens = []
#     i = 0
#     while (i < len(tokens)):
#         if (((tokens[i] == "N") or (tokens[i] == "A")) and (i < (len(tokens) - 1)) and (tokens[i+1] in allowed_cases)):
#             new_tokens.append(tokens[i] + " " + tokens[i+1])
#             i += 2
#         else:
#             new_tokens.append(tokens[i])
#             i += 1
#     return new_tokens

# Noise.add_noise(context_tokens("A ने A को A से"))
# Noise.add_noise(context_tokens("A ने A से A को"))
# Noise.add_noise(context_tokens("A को A ने A से"))
# Noise.add_noise(context_tokens("A को A से A ने"))
# Noise.add_noise(context_tokens("A से A को A ने"))
# Noise.add_noise(context_tokens("A से A ने A को"))

# Noise.add_noise(context_tokens("A को A ने A को"))
# Noise.add_noise(context_tokens("A को A से A को"))
# Noise.add_noise(context_tokens("A से A ने A से"))
# Noise.add_noise(context_tokens("A से A को A से"))