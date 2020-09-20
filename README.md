## Differentiating Metal-Poor Main Sequence Stars from White Dwarfs with Spectro-Photometric Observables

**Tutorial**: [Demo Notebook](https://dfm.io/nbview/?url=https%3A%2F%2Fgithub.com%2Fvedantchandra%2Fmpms%2Fblob%2Fmaster%2F0_demo.ipynb)

**Read the Paper**: [(placeholder link)](https://vedantchandra.com)

This repository is an accompaniment to Chandra & Schlaufman (2020, in prep.). The contents are as follows:

- `/paper/` contains all notebooks required to reproduce the results of the paper, minus large files for the atmospheric model grids. These files can be obtained from the [PHOENIX website (MPMS models)](http://phoenix.astro.physik.uni-goettingen.de/) and from [Simon Blouin (WD models)](https://www.lanl.gov/search-capabilities/profiles/simon-blouin.shtml) respectively. 

-  `classify_mpms.py` contains the code deliverable from our paper - a way for anyone to use our logistic regression classifier to differentiate metal-poor main-sequence stars from white dwarfs on the basis of spectro-photometric observables. 

- `training_grid.csv` contains our grid of training features, derived from the latest synthetic spectra for MPMS and WD stars. For more details, refer to the paper. The classifier trains itself using this grid when initialized. Others are free to use this grid directly to train their own classification algorithm. 

- `0_demo.ipynb` is a simple demonstration of the methods described in the paper. It describes how one can use the classes and functions in `classify_mpms.py` to go from an observed spectrum (and/or photometric observations) to a computed probability of a given star being an MPMS candidate or a white dwarf. 

#### Non-Standard Python Dependancies:

- `lmfit`
- `scikit-learn`
- `pandas`
