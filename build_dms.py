import numpy as np
import utils
import pickle
from urllib.request import urlretrieve
import zipfile

tol = 1e-8
phrasetypes = ['SV', 'VO', 'SVO']
normalisation='maxeig1'

print("Starting download")
glove_zip, headers = urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip')
print("Download done")

with zipfile.ZipFile(glove_zip,"r") as zip_ref:
    zip_ref.extractall("glove")
vectorfile = 'glove/glove.6B.50d.txt'

print("Loading noun hyponym dictionary")
with open("all-hypos.p", 'rb') as hyp_file:    
    hyponyms = pickle.load(hyp_file)   
print("Done.")

print("Getting vectors for each hyponym")
vectors = utils.get_vectors(hyponyms, vectorfile)
print("Done.")
print("Building density matrices")
density_matrices = utils.build_density_matrices(hyponyms, vectors, normalisation=normalisation)
print("Done.")
print("We built {0} density matrices".format(len(density_matrices)))
print("Pickling density matrices")
with open("dm-50d-glove-wn.p", "wb" ) as dm_file:
    pickle.dump(density_matrices, dm_file)
print("Done.")
