import numpy as np
from scipy.stats import ttest_1samp, ttest_rel

import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="the input filename")
args = parser.parse_args()
filename = args.filename

ksr = [.84]

with open(filename, 'rb') as f:
    empirical = pickle.load(f, encoding='bytes')

for k in empirical:
    print(f'{k}: {np.mean(empirical[k])}\n')


for k in empirical:
    if k[0] == 0:
        tp = ttest_1samp(empirical[k], ksr[0])
        print('{0} {1}, mean {2:.4f}'.format(k, tp, np.mean(empirical[k])))

print('Comparison with kraus')
for k in empirical:
    tp = ttest_rel(empirical[k], empirical[(k[0], 'kraus')])
    print('{0} ts: {1:.5f}, p: {2:.3g}'.format(k, tp[0], tp[1]))
        
print('Comparison with mult')
for k in empirical:
    tp = ttest_rel(empirical[k], empirical[(k[0], 'mult')])
    print('{0} ts: {1:.5f}, p: {2:.3g}'.format(k, tp[0], tp[1]))
    
print('Comparison with noun_only')
for k in empirical:
    tp = ttest_rel(empirical[k], empirical[(k[0], 'project')])
    print('{0} ts: {1:.5f}, p: {2:.3g}'.format(k, tp[0], tp[1]))
