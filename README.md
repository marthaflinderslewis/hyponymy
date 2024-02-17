# Density Matrices for Hyponymy and Entailment

This code forms the basis for papers https://aclanthology.org/R19-1075/, https://arxiv.org/abs/2005.04929, and https://arxiv.org/abs/2005.14134.

# Requirements
nltk
numpy
scikit-learn


# Instructions
First run `get_hyponyms.py` to get hyponyms from WordNet (requires `nltk`).

Then run `build_dms` to build density matrices using hyponyms collected from WordNet and GloVe vectors. Code uses GloVe 50d but this can be changed.

Run `compose.py` to test density matrices on the KS2016 dataset with negation.

`stats_neg.py` calculates statistical significance of results.

# Contact details
Please look at https://marthaflinderslewis.github.io/ for up to date contact details.