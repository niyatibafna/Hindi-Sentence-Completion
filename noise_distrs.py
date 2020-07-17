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
            else:
                return 0
        else:
            return (count(c, r) * self.prob_context(r) * (self.prob_red_csize ** (len(c) - len(r))))

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
            for r in csize_rs:
                r_prob = (self.prob_context(r) * d_pen)/self.total_prob_csize[csize]
                all_rs_c.append(r)
                prob_rs_c.append(r_prob)
            if (np.random.uniform(0, 1) > self.prob_red_csize):
                break
        if (len(all_rs_c) == 1):
            return (all_rs_c[0] + chosen_r)
        else:
            inds = sorted(range(len(prob_rs_c)), key=lambda x: prob_rs_c[x], reverse=True)[:self.topk]
            all_rs_c = [all_rs_c[i] for i in inds]
            prob_rs_c = [prob_rs_c[i] for i in inds]
            prob_rs_c = normalize(prob_rs_c)
            print(list(zip(all_rs_c, prob_rs_c)))
            return (all_rs_c[np.random.choice(np.arange(self.topk), p=prob_rs_c)] + chosen_r)