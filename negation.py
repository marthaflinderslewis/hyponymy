import numpy as np
import scipy as sc
import utils as ent
import random
import pickle
from sklearn.metrics import roc_auc_score

# random.seed(0)
path_root = '/Users/martha/'

normalisation='maxeig1' #must use maxeig1 normalization
vectorfile = path_root + 'data/glove.6B/glove.6B.50d.txt'

d=50
tol = 1e-8
pt = 'SV'
datapath = path_root + 'data/KS2016/KS2016-'
# combination_funcs = [ent.verb_only, ent.noun_only, ent.average, ent.mult, ent.kraus, ent.project, ent.geom_mean, ent.parallel_mean]
combination_funcs = [ent.verb_only, ent.noun_only, ent.average, ent.mult, ent.kraus, ent.project]
combination_funcs = [ent.traced_verb_only,  ent.traced_noun_only, ent.diag, 
                        ent.sum_verb_only,  ent.sum_noun_only,
                        ent.sum_n_diag_v,  ent.sum_v_diag_n, ent.traced_addition,
                        ent.mult, ent.project, ent.kraus]
entailment_func = ent.kBA
rocauc = {}


datafile = datapath + pt + '.txt'
print("Getting phrase pairs and entailments values...")
phrase_pairs = ent.get_ks2016(datafile)
print("Done.")

plain_pp = phrase_pairs
nv_pp = {(pp[0], pp[3], pp[2], pp[1]): v for pp, v in phrase_pairs.items()}
nn_pp = {(pp[2], pp[1], pp[0], pp[3]): v for pp, v in phrase_pairs.items()}
both_pp = {(pp[2], pp[3], pp[0], pp[1]): v for pp, v in phrase_pairs.items()}

quantified_phrase_pairs = [plain_pp, nv_pp, nn_pp, both_pp]
qpp_names = ['plain', 'neg_verb', 'neg_noun', 'both']

print("Opening the pickled density matrices...")
density_matrices = pickle.load(open(path_root + "data/KS2016/May2019/ks2016-dm-all-50d-glove-wn.p", "rb" ))
print("Done.")

density_matrices = {k.lower(): ent.normalize(v, normalisation) for k, v in density_matrices.items()}
# density_matrices = {k.lower(): ent.normalize(v + 0.5*np.eye(d), normalisation) for k, v in density_matrices.items()}

for (name, qp) in zip(qpp_names, quantified_phrase_pairs):
    for f in combination_funcs:
        print("Calculating values for {0}, {1}...".format(name, f.__name__))
        combined_phrases = ent.calc_neg_entailments(qp, name, d, density_matrices, f, pt, \
            norm_type=normalisation, entailment_func=entailment_func, verb_operator=True)
        rocauc = ent.calculate_roc_bootstraps(qp, combined_phrases, name, \
            rocauc, f.__name__)
#         rocauc = ent.calculate_roc(qp, combined_phrases, name, \
#             rocauc, f.__name__)
        if f.__name__ == 'project' or f.__name__ == 'kraus':
            combined_phrases = ent.calc_neg_entailments(qp, name, d, density_matrices, f, \
                    pt, norm_type=normalisation, entailment_func=entailment_func, \
                    verb_operator=False)
            rocauc = ent.calculate_roc_bootstraps(qp, combined_phrases, \
                    name, rocauc, f.__name__+'_switched')
#             rocauc = ent.calculate_roc(qp, combined_phrases, \
#                     name, rocauc, f.__name__+'_switched')
                

    print("Done")
    
print("Writing results to text file...")
with open("results-{0}d-glove-negation-{1}-{2}.txt".format(d,entailment_func.__name__, normalisation), 'w') as outfile:
    for k in sorted(rocauc.keys()):
        outfile.write(' '.join(map(str, k))+' '+str(rocauc[k])+'\n')
print("Done.")
        
print("Saving out bootstraps")
print("pickling out results...")
with open("results_{0}d-negation-new-{1}.p".format(d, entailment_func.__name__), 'wb') as f:
    pickle.dump(rocauc, f)
print("Done")

