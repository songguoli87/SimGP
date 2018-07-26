# SimGP

Matlab code for Multimodal Similarity GPLVMs (G.Song, et al., "Similarity Gaussian Process Latent Variable Model
for Multi-Modal Data Analysis", ICCV 2015; "Multimodal Similarity Gaussian Process Latent
Variable Model", TIP 2017)

# Prerequisites
(1) GPmat - Neil Lawrence's GP matlab toolbox: https://github.com/SheffieldML/GPmat

(2) Netlab v.3.3: http://www1.aston.ac.uk/ncrg/

# Data
pascal1K.mat: original feature (SIFT for image and LDA for text) 

pascal1k_similarity_euc: similarity feature generated by running makedata.m

# Models
(1) main_rsimgp - An implementation for m-SimGP and m-RSimGP

- m-SimGP: options_y.prior and options_z.prior are Gaussian prior;
                  
- m-RSimGP: options_y.prior = [], options_z.prior = [], and running sgplvmAddConstraint(model,options_constraint) with                       options_constraint = constraintOptions('Sim').

(2) main_drsimgp - An implementation for m-DSimGP and m-DRSimGP

- m-DSimGP: options.neighborconstraints = true; 

- m-DRSimGP: options.neighborconstraints = true, and options_constraint = constraintOptions('Sim').


# Evaluation
Evaluation of mAP for the cross-modal retrieval task is implemented in the retrieval.m file.
