**Digital Twins for Counterfactual Analysis in M4EFAD Study
Makes up final results section of Optimising recovery from childhood moderate acute malnutrition: a randomized controlled trial**

Scripts contains the code for this analysis.
1. **save_microbiome.py** was used to do PCA on the microbiome data
2. **format_matrix_og.py** combined all datasets
3. **vae.py** was used to create a variational auto-encoder to represent the data. Model output in **models**
4. **outcome_heads_nobrain.py** was used to train regression neural networks on top of the VAE from 3. Model output in **models**
5. **what_if_counterfactuals.py** was used to run counterfactual analyses (e.g. what personalised changes are required to bring everyone up to a weight-for-height of x?)
6. **make_figs.py** was used to make figures from the output of what_if_counterfactuals.py. Some of these (the ones in the paper) are in **outcomes**
