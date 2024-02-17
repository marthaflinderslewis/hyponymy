import numpy as np
import scipy as sc
import utils
import random
import pickle
from sklearn.metrics import roc_auc_score

random.seed(0)

normalisation = 'maxeig1'
vectorfile = 'glove.6B.50d.txt'

d=50
tol = 1e-8
pt = 'SV'
datafile = 'data/KS2016-SV.txt'
combination_funcs = [utils.verb_only,  utils.addition, utils.mult, utils.project, utils.kraus]
# for 'Cats climb entails mammals move', use the combination functions below:
# combination_funcs = [utils.traced_verb_only,  utils.traced_noun_only, utils.diag, 
#                         utils.sum_verb_only,  utils.sum_noun_only,
#                         utils.sum_n_diag_v,  utils.sum_v_diag_n, utils.traced_addition,
#                         utils.mult, utils.project, utils.kraus]
entailment_func = utils.kE

rocauc = {}

print("Getting phrase pairs and entailments values...")
phrase_pairs = utils.get_ks2016(datafile)
print("Done.")

plain_pp = phrase_pairs
nv_pp = {(pp[0], pp[3], pp[2], pp[1]): v for pp, v in phrase_pairs.items()}
nn_pp = {(pp[2], pp[1], pp[0], pp[3]): v for pp, v in phrase_pairs.items()}
both_pp = {(pp[2], pp[3], pp[0], pp[1]): v for pp, v in phrase_pairs.items()}

quantified_phrase_pairs = [plain_pp, nv_pp, nn_pp, both_pp]
qpp_names = ['plain', 'neg_verb', 'neg_noun', 'both']

print("Opening the pickled density matrices...")
with open("dm-50d-glove-wn.p", "rb") as dm_file:
    density_matrices = pickle.load(dm_file)
print("Done.")
density_matrices = {k.lower(): utils.normalize(v, normalisation) for k, v in density_matrices.items()}

for k, v in density_matrices.items():
    if max(np.linalg.eigvalsh(v)) > 1:
        print("{0}, {1}".format(k, max(np.linalg.eigvalsh(v))))
        
for (name, qp) in zip(qpp_names, quantified_phrase_pairs):
    for f in combination_funcs:
        print("Calculating values for {0}, {1}...".format(name, f.__name__))
        combined_phrases = utils.calc_neg_entailments(qp, name, d, density_matrices, f, pt, \
            norm_type=normalisation, entailment_func=entailment_func, verb_operator=True)
        rocauc = utils.calculate_roc_bootstraps(qp, combined_phrases, name, \
            rocauc, f.__name__)
        if f.__name__ == 'project' or f.__name__ == 'kraus':
            combined_phrases = utils.calc_neg_entailments(qp, name, d, density_matrices, f, \
                    pt, norm_type=normalisation, entailment_func=entailment_func, \
                    verb_operator=False)
            rocauc = utils.calculate_roc_bootstraps(qp, combined_phrases, \
                    name, rocauc, f.__name__+'_switched')
    print("Done")

print("Writing results to text file...")
with open(f"results-50d-glove-{entailment_func.__name__}-{normalisation}.txt", 'w') as outfile:
    for k in sorted(rocauc.keys()):
        outfile.write(' '.join(map(str, k))+' '+str(rocauc[k])+'\n')
print("Done.")
        
print("Saving out bootstraps")
print("pickling out results...")
with open(f"results-50d-glove-{entailment_func.__name__}-{normalisation}.p", 'wb') as f:
    pickle.dump(rocauc, f)
print("Done")














