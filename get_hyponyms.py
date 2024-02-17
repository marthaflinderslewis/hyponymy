import numpy as np
import utils
import pickle

tol = 1e-10
phrasetypes = ['SV', 'VO', 'SVO']
datapath = 'data/KS2016-'
words = []

for pt in phrasetypes:
    datafile = datapath + pt + '.txt'
    print("Getting phrase pairs and entailments values for phrasetype {0}...".format(pt))
    phrase_pairs = utils.get_ks2016(datafile)
    print("Done.")
    print("Creating vocab list from phrase pairs...")
    words += utils.get_vocab(list(phrase_pairs.keys()))
    print("Done")
print("Traversing WordNet hierarchy to get hyponyms...")
hyponyms = utils.get_hyponyms(words)
print("Done.")
print("Pickling out dictionary of hyponyms...")
with open("all-hypos.p", "wb" ) as outfile:
    pickle.dump(hyponyms, outfile)
print("Done.")

