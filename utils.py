import numpy as np
# import scipy as sc
# from itertools import islice
from nltk.corpus import wordnet as wn
import random
# from functools import reduce
# import operator
# import re
# import collections
# import copy
# import time
# import sys
# from collections import deque
# from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import math

# get_ks2016 reads in the KS2016 dataset and returns a dictionary with the sentence pairs
# as keys (as a tuple of words) and the entailment value as value (either T or F)
# Format:
# evidence suggest,information express,T
def get_ks2016(file):
    scores = {}
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(',')
            entries = [word.lower() for phrase in tmp for word in phrase.split()]
            scores[tuple(entries[:-1])] = entries[-1]
    return scores
 
# get_vocab takes a dict of phrase pairs and scores and returns a list of individual words
def get_vocab(scores):
    word_list = [word for phrases in scores for word in phrases]
    vocab = list(set(word_list))
    return vocab
    
def get_vector_vocab(vfile):
    vocab = []
    with open(vfile, 'r') as f:
        for line in f:
            vocab.append(line.split()[0])
    return set(vocab)

#function to get hyponyms
def hypo(s):
    return s.hyponyms()

# get_hyponyms takes a list of words, an optional part-of-speech tag
# (default is None), and an optional depth (default 10),
# and returns a dictionary whose keys are the words in the list and
# whose values are lists of hyponyms of each word, taken from the transitive
# closure of WordNet to the depth specified
def get_hyponyms(word_list, pos=None, depth=10):
    hyponyms = {word:[] for word in word_list}
    count = 0
    for word in word_list:
        count += 1
        if count % 100 == 0:
            print("Got the hyponyms of {0} words out of {1}".format(count, len(word_list)))
        #get synsets of word
        synset_list = wn.synsets(word, pos=pos)
        if len(synset_list) > 0:
            for synset in synset_list:
                # collect all the synsets below a given synset
                synsets = list(synset.closure(hypo, depth=depth))
                # include the synset itself as well
                synsets.append(synset)
                for s in synsets:
                    for ln in s.lemma_names():
                        hyponyms[word].append(ln.lower())
            hyponyms[word] = list(set(hyponyms[word]))
        else:
            hyponyms[word] = 'OOV'
    return hyponyms

# get_vectors takes a dictionary of words and their hyponyms, and a file
# containing vectors, and returns a dictionary with vocab as keys and vectors as values
# vector file has format
# word val val ... val
def get_vectors(hyp_dict, vec_file, normalisation=False, weights=False, header=False):
    if weights:
        vocabulary = set([hyp[0] for word in hyp_dict for hyp in hyp_dict[word] if hyp_dict[word] != 'OOV'])
    else: 
        vocabulary = set([hyp for word in hyp_dict for hyp in hyp_dict[word] if hyp_dict[word] != 'OOV'])
    hypo_vectors = {}
    with open(vec_file, 'r') as vf:
        if header:
            vf.readline()
        for line in vf:
            entry = line.split()
            if entry[0] in vocabulary:
                vec = np.array([float(n) for n in entry[1:]])
                if normalisation:
                    vec = vec/np.linalg.norm(vec)
                hypo_vectors[entry[0]] = vec
    return hypo_vectors

def build_density_matrices(hyp_dict, hypo_vectors, normalisation=False, weights=False):
    dim = len(random.choice(list(hypo_vectors.values()))) #dim= length of arbitrary vector
    vocab = list(hyp_dict.keys())
    vectors = {word:np.zeros([dim, dim]) for word in vocab}
    not_found = {word:[] for word in vocab}
    zeros = []
    for word in hyp_dict:
        # TODO: deal with warning
        if hyp_dict[word] == 'OOV':
            continue
        for hyp in hyp_dict[word]:
            if hyp not in hypo_vectors:
                continue
            v = hypo_vectors[hyp] #make sure I alter the Hearst code
            vv = np.outer(v, v)
            vectors[word] += vv
        v = vectors[word]
        if np.all(v == 0):
            vectors[word] = 'OOV'
            continue
        if normalisation == 'trace1':
            assert np.trace(v) != 0, "Trace is 0, should be OOV"
            v = v/np.trace(v)
            vectors[word] = v
        elif normalisation == 'maxeig1':
            maxeig = np.max(np.linalg.eigvalsh(v))
            assert maxeig != 0, "Max eigenvalue is 0, should be OOV"
            v = v/maxeig
            vectors[word] = v
        elif not normalisation:
            pass
        else:
            print('Possible arguments to normalisation are "trace1", "maxeig1" or False (default).  You entered {0}'.format(normalisation))
            break
    return vectors

# Combination methods

def compose(matlist, f, norm_type=None, **kwargs):
    mat = f(matlist, **kwargs)
    mat = normalize(mat, norm_type)
    return mat
    
def verb_only(matlist, pt = 'SV', **kwargs):
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt) 
    if pt == 'SV' or pt == 'SVO':
        place = 1
    else:
        place = 0
    mat = matlist[place]
    return mat
    
def traced_verb_only(matlist, pt = 'SV', normalisation = 'maxeig1', **kwargs):
    if normalisation == 'maxeig1':
        dim = np.shape(matlist[0])[0]
    else: 
        dim = 1.
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt) 
    if pt == 'SV':
        mat = matlist[1]*np.trace(matlist[0])/dim
    elif  pt == 'SVO':
        mat = matlist[1]*np.trace(matlist[0])*np.trace(matlist[2])/(dim*dim)
    else:
        mat = matlist[0]*np.trace(matlist[1])/dim
    return mat
    
def noun_only(matlist, pt = 'SV', **kwargs):
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt) 
    if pt == 'SV' or pt == 'SVO':
        place = 0
    else:
        place = 1
    mat = matlist[place]
    return mat
    
def traced_noun_only(matlist, pt = 'SV', normalisation = 'maxeig1', **kwargs):
    if normalisation == 'maxeig1':
        dim = np.shape(matlist[0])[0]
    else: 
        dim = 1.
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt) 
    if pt == 'SV':
        mat = matlist[0]*np.trace(matlist[1])/dim
    elif  pt == 'SVO':
        mat = matlist[0]*np.trace(matlist[1])*np.trace(matlist[2])/(dim*dim)
    else:
        mat = matlist[1]*np.trace(matlist[0])/dim
    return mat
    
def addition(matlist, **kwargs):
    mat = sum(matlist)
    return mat
    
def sum_verb_only(matlist, pt = 'SV', normalisation = 'maxeig1', **kwargs):
    if normalisation == 'maxeig1':
        dim = np.shape(matlist[0])[0]
    else: 
        dim = 1.
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt) 
    if pt == 'SV':
        mat = matlist[1]*np.sum(matlist[0])/(dim**2)
    elif  pt == 'SVO':
        mat = matlist[1]*np.sum(matlist[0])*np.sum(matlist[2])/(dim**4)
    else:
        mat = matlist[0]*np.sum(matlist[1])/dim
    return mat
    
def sum_noun_only(matlist, pt = 'SV', normalisation = 'maxeig1', **kwargs):
    if normalisation == 'maxeig1':
        dim = np.shape(matlist[0])[0]
    else: 
        dim = 1.
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt) 
    if pt == 'SV':
        mat = matlist[0]*np.sum(matlist[1])/(dim**2)
    elif  pt == 'SVO':
        mat = matlist[0]*np.sum(matlist[1])*np.sum(matlist[2])/(dim**4)
    else:
        mat = matlist[1]*np.sum(matlist[0])/(dim**2)
    return mat
    
def addition(matlist, **kwargs):
    mat = sum(matlist)
    return mat
        
def traced_addition(matlist, **kwargs):
    dim = np.shape(matlist[0])[0]
    assert len(matlist) == 2 or len(matlist) == 3, \
            "unexpected list length: should be 2 for SV and VO, 3 for SVO. You have \
             {0}".format(len(matlist))
    if len(matlist) == 2:
        mat = (np.trace(matlist[0])*matlist[1] + np.trace(matlist[1])*matlist[0])/2
        mat = mat/dim
        return mat
    if len(matlist) == 3:
        mat = (np.trace(matlist[1])*matlist[2] + np.trace(matlist[2])*matlist[1])/2
        mat = mat/dim
        mat2 = (np.trace(mat)*matlist[0] + np.trace(matlist[0])*mat)/2
        mat2 = mat2/dim
        return mat2
        
def diag(matlist, **kwargs):
    mat = np.diag(np.diag(matlist[0]))
    for m in matlist[1:]:
        mat = mat.dot(np.diag(np.diag(m)))
    return mat
    
def tad(matlist, **kwargs):
    dim = np.shape(matlist[0])[0]
    assert len(matlist) == 2 or len(matlist) == 3, \
            "unexpected list length: should be 2 for SV and VO, 3 for SVO. You have \
             {0}".format(len(matlist))
    if len(matlist) == 2:
        mat = (np.trace(matlist[0])*matlist[1] + np.trace(matlist[1])*matlist[0])/dim \
                    + diag(matlist)
        mat = mat/3
        return mat
    if len(matlist) == 3:
        mat = (np.trace(matlist[1])*matlist[2] + np.trace(matlist[2])*matlist[1])/dim \
                    + diag(matlist[1:3])
        mat = mat/3
        mat2 = (np.trace(mat)*matlist[0] + np.trace(matlist[0])*mat)/dim \
                    + diag([matlist[0], mat])
        mat2 = mat2/3
        return mat2
    
def sum_addition(matlist, **kwargs):
    dim = np.shape(matlist[0])[0]
    assert len(matlist) == 2 or len(matlist) == 3, \
            "unexpected list length: should be 2 for SV and VO, 3 for SVO. You have \
             {0}".format(len(matlist))
    if len(matlist) == 2:
        mat = (np.sum(matlist[0])*matlist[1] + np.sum(matlist[1])*matlist[0])/2
        mat = mat/(dim**2)
        return mat
    if len(matlist) == 3:
        mat = (np.sum(matlist[1])*matlist[2] + np.sum(matlist[2])*matlist[1])/2
        mat = mat/(dim**2)
        mat2 = (np.sum(mat)*matlist[0] + np.sum(matlist[0])*mat)/2
        mat2 = mat2/(dim**2)
        return mat2
        
def sum_n_diag_v(matlist, pt = 'SV', normalisation = 'maxeig1', **kwargs):
    if normalisation == 'maxeig1':
        dim = np.shape(matlist[0])[0]
    else: 
        dim = 1.
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt) 
    if pt == 'SV':
        mat = np.diag(np.diag(matlist[1]))*np.sum(matlist[0])/(dim**2)
    elif  pt == 'SVO':
        #when we iterate compr we end up summing over the diagonal, which is the trace
        mat = np.diag(np.diag(matlist[1]))*np.sum(matlist[0])*np.trace(matlist[2])/(dim**3)
    else:
        mat = np.diag(np.diag(matlist[0]))*np.sum(matlist[1])/(dim**2)
    return mat
    
def sum_v_diag_n(matlist, pt = 'SV', normalisation = 'maxeig1', **kwargs):
    if normalisation == 'maxeig1':
        dim = np.shape(matlist[0])[0]
    else: 
        dim = 1.
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt) 
    if pt == 'SV':
        mat = np.diag(np.diag(matlist[0]))*np.sum(matlist[1])/(dim**2)
    elif  pt == 'SVO':
        mat = np.diag(np.diag(matlist[0]))*np.sum(matlist[1])*np.trace([matlist[2]])/(dim**3)
    else:
        mat = np.diag(np.diag(matlist[1]))*np.sum(matlist[0])/(dim**2)
    return mat
                
    
def average(matlist, **kwargs):
    mat = sum(matlist)/len(matlist)
    return mat
    
def mult(matlist, **kwargs):
    mat = matlist[0]
    for m in matlist[1:]:
        mat = np.multiply(mat, m)
    return mat
    
    
def matsqrt(A, tol=1e-8):
    dim = np.shape(A)[0]
    assert np.all(np.abs(A.imag) < tol), "parts of A complex"
    A = np.real(A)
    assert check_symmetric(A), "A is not symmetric"
    vals, vecs = np.linalg.eigh(A)
    assert np.all(np.abs(vals.imag) < tol), "Some eigenvalues complex"
    vals = np.real(vals)
    vecs = np.array([vec for val, vec in  zip(vals, vecs) if np.abs(val) > tol])
    vals = vals[np.abs(vals) > tol]
    assert np.all(vals >= 0.), vals
    sqrtvals = [math.sqrt(v) for v in vals]
    assert len(vecs) == len(sqrtvals), "different number of eigenvectrs and values"
    assert np.all(np.abs(vecs.imag) < tol), "Some eigenvectors complex: {0}".format(vecs)
    vecs = np.real(vecs)
#     print(vecs)
#     print(sqrtvals)
    sqrtA = vecs.T.dot(np.diag(sqrtvals)).dot(vecs)
    return sqrtA
    
     
def project(matlist, verb_operator=True, pt='SV', tol=1e-8, **kwargs):
    assert len(matlist) == 2 or len(matlist) == 3, \
            "unexpected list length: should be 2 for SV and VO, 3 for SVO. You have \
             {0}".format(len(matlist))
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt)
    if pt == 'SV':
        noun = matlist[0]
        verb = matlist[1]
        if verb_operator:
            verbsqrt = matsqrt(verb)
            assert np.all(np.abs(verbsqrt.imag) < tol), "parts of verb complex"
            mat = verbsqrt.dot(noun).dot(verbsqrt)
        else:
            nounsqrt = matsqrt(noun)
            assert np.all(np.abs(nounsqrt.imag) < tol), "parts of noun complex"
            mat = nounsqrt.dot(verb).dot(nounsqrt)
    elif pt == 'VO':
        noun = matlist[1]
        verb = matlist[0]
        if verb_operator:
            verbsqrt = matsqrt(verb)
            assert np.all(np.abs(verbsqrt.imag) < tol), "parts of verb complex"
            mat = verbsqrt.dot(noun).dot(verbsqrt)
        else:
            nounsqrt = matsqrt(noun)
            assert np.all(np.abs(nounsqrt.imag) < tol), "parts of noun complex"
            mat = nounsqrt.dot(verb).dot(nounsqrt)
    else: # pt == 'SVO'
        noun1 = matlist[0]
        verb = matlist[1]
        noun2 = matlist[2]
        if verb_operator:
            verbsqrt = matsqrt(verb)
            assert np.all(np.abs(verbsqrt.imag) < tol), "parts of verb complex"
            mat = verbsqrt.dot(noun2).dot(verbsqrt)
            msqrt = matsqrt(mat)
            assert np.all(np.abs(msqrt.imag) < tol), "parts of mat complex"
            mat = msqrt.dot(noun1).dot(msqrt)
        else:
            noun2sqrt = matsqrt(noun2)
            noun1sqrt = matsqrt(noun1)
            mat = noun2sqrt.dot(verb).dot(noun2sqrt)
            mat = noun1sqrt.dot(mat).dot(noun1sqrt)
    return mat

def check_symmetric(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)

# helper function for kraus     
def kmult(A, B, tol=1e-8):
    assert np.all(np.abs(A.imag) < tol), "parts of A complex"
    A = np.real(A)
    assert np.all(np.abs(B.imag) < tol), "parts of B complex"
    B = np.real(B)
    assert check_symmetric(A), "A is not symmetric"
    vals, vecs = np.linalg.eigh(A)
    assert np.all(np.abs(vals.imag) < tol), "Some eigenvalues complex"
    vals = np.real(vals)
    vecs = np.array([vec for val, vec in  zip(vals, vecs) if np.abs(val) > tol])
    vals = vals[np.abs(vals) > tol]
    assert len(vecs) == len(vals)
    assert np.all(np.abs(vecs.imag) < tol), "Some eigenvectors complex: {0}".format(vecs)
    vecs = np.real(vecs)
    vals_projectors = []
    AB = np.zeros(np.shape(A))
    for vec, val in zip(vecs, vals):
        p = np.outer(vec, vec)
        AB += val*p.dot(B.dot(p))
    return AB        
  
def kraus(matlist, verb_operator=True, pt='SV', **kwargs):
    assert len(matlist) == 2 or len(matlist) == 3, \
            "unexpected list length: should be 2 for SV and VO, 3 for SVO. You have \
             {0}".format(len(matlist))
    assert pt in ['SV', 'VO', 'SVO'], "Unexpected phrasetype: should be 'SV', 'VO', \
    'SVO'. You have {0}".format(pt)
    if pt == 'SV':
        noun = matlist[0]
        verb = matlist[1]
        if verb_operator:
            mat = kmult(verb, noun)
        else:
            mat = kmult(noun, verb)
    elif pt == 'VO':
        noun = matlist[1]
        verb = matlist[0]
        if verb_operator:
            mat = kmult(verb, noun)
        else:
            mat = kmult(noun, verb)
    else: #pt == 'SVO'
        noun1 = matlist[0]
        verb = matlist[1]
        noun2 = matlist[2]
        if verb_operator:
            mat = kmult(verb, noun2)
            mat = kmult(mat, noun1)
        else:
            mat = kmult(noun2, verb)
            mat = kmult(noun1, mat)
    return mat
    
#matrix normalization function
def normalize(A, norm_type):
    assert norm_type in ['trace1', 'maxeig1', None], \
        'Possible arguments to normalize are "trace1", "maxeig1" or None (default). \
        You entered "{0}".'.format(norm_type)
    if norm_type == 'trace1':
        assert np.trace(A) != 0, 'Trace of A is 0, cannot normalize'
        A = A/np.trace(A)
    elif norm_type == 'maxeig1':
        maxeig = np.max(np.linalg.eigvalsh(A))
        if maxeig >= 1:
#             print("maxeig was greater than 1: normalizing, {0}".format(maxeig))
            A = A/maxeig
    return A

#entailment functions
def symb(Aword, Bword, hyp_dict):
    if Aword in hyp_dict[Bword]:
        return 1
    else:
        return 0

def kE(A, B, tol=1e-8):
    assert not np.all(A == 0) and not np.all(B == 0), "Zero matrix found"
    eigvals = np.linalg.eigvalsh(B - A)
    assert np.all(np.abs(eigvals.imag) < tol), "Some eigenvalues complex"
    eigvals = eigvals.real
    E = np.diag([e if e < 0 else 0 for e in eigvals])
    return 1-np.linalg.norm(E)/np.linalg.norm(A)

def kBA(A, B, tol=1e-8):
    assert not np.all(A == 0) and not np.all(B == 0), "Zero matrix found"
    eigvals = np.linalg.eigvalsh(B - A)
    assert np.all(np.abs(eigvals.imag) < tol), "Some eigenvalues complex"
    eigvals = eigvals.real
    if np.all(eigvals == 0):
        return 1
    else:
        return sum(eigvals)/sum(np.abs(eigvals))
        
def calc_entailments(phrase_pairs, density_matrices, f, pt, norm_type='maxeig1', entailment_func=kBA, verb_operator=True, tol=1e-8):
    combined_phrases = {}
    for phrase_pair in phrase_pairs:
        wv_list = [density_matrices[p] for p in phrase_pair]
        midpt = int(len(wv_list)/2)
        w_list = wv_list[:midpt]
        v_list = wv_list[midpt:]
        w = compose(w_list, f, pt=pt, verb_operator=verb_operator, norm_type=norm_type)
        v = compose(v_list, f, pt=pt, verb_operator=verb_operator, norm_type=norm_type)
        combined_phrases[phrase_pair] = entailment_func(w, v, tol)
    return combined_phrases
    
def calc_neg_entailments(phrase_pairs, i, d, density_matrices, f, pt, norm_type='maxeig1', entailment_func=kBA, verb_operator=True, tol=1e-8):
    combined_phrases = {}
    for phrase_pair in phrase_pairs:
        if i == 'plain':
            wv_list = [density_matrices[p] for p in phrase_pair]
        elif i == 'neg_verb':
            wv_list = [density_matrices[phrase_pair[0]], \
                        np.eye(d) - density_matrices[phrase_pair[1]],\
                        density_matrices[phrase_pair[2]], \
                        np.eye(d) - density_matrices[phrase_pair[3]]]
        elif i == 'neg_noun': 
            wv_list = [np.eye(d) - density_matrices[phrase_pair[0]], \
                        density_matrices[phrase_pair[1]],\
                         np.eye(d) - density_matrices[phrase_pair[2]], \
                         density_matrices[phrase_pair[3]]]
        else:
            wv_list = [np.eye(d) - density_matrices[phrase_pair[0]], \
                        np.eye(d) - density_matrices[phrase_pair[1]],\
                         np.eye(d) - density_matrices[phrase_pair[2]], \
                         np.eye(d) - density_matrices[phrase_pair[3]]]
                         
        midpt = int(len(wv_list)/2)
        w_list = wv_list[:midpt]
        v_list = wv_list[midpt:]
        w = compose(w_list, f, pt=pt, verb_operator=verb_operator, norm_type=norm_type)
        v = compose(v_list, f, pt=pt, verb_operator=verb_operator, norm_type=norm_type)
        combined_phrases[phrase_pair] = entailment_func(w, v, tol)
    return combined_phrases
    
def calculate_roc(phrase_pairs, combined_phrases, pt, roc_dict, func_name):
    sorted_true = [phrase_pairs[key] for key in sorted(phrase_pairs.keys())]
    sorted_calculated = [combined_phrases[key] for key in sorted(phrase_pairs.keys())]
    sorted_true = [1 if val == 't' else 0 for val in sorted_true]
    roc_dict[(pt, func_name)] = roc_auc_score(sorted_true, sorted_calculated)
    return roc_dict

def calculate_roc_bootstraps(phrase_pairs, combined_phrases, pt, roc_dict, func_name):
    for i in range(100):
        bs_keys = [random.choice(list(phrase_pairs.keys())) for _ in range(len(phrase_pairs))]
        sorted_true = [phrase_pairs[key] for key in sorted(bs_keys)]
        sorted_calculated = [combined_phrases[key] for key in sorted(bs_keys)]
        sorted_true = [1 if val == 't' else 0 for val in sorted_true]
        if (pt, func_name) in roc_dict:
            roc_dict[(pt, func_name)].append(roc_auc_score(sorted_true, \
                                                                sorted_calculated))
        else: 
            roc_dict[(pt, func_name)] = [roc_auc_score(sorted_true, sorted_calculated)]
    return roc_dict
    

if __name__ == "__main__":
	pass
