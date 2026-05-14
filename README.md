# Neural transfer of biophysical generating-function solutions enables robust integration of multinomial single-cell sequencing data


## Introduction

Multinomial single-cell sequencing data are often integrated with feature-concatenation strategies that can be effective empirically but do not explicitly preserve the biophysical structure of gene-expression dynamics. Here we present BRIDGE, a biophysically grounded neural framework that maps the known probability generating function (PGF) of an analytically tractable single-species gene-expression model to the unknown PGF of a larger multivariate model that contains it. This repository (**BRIDGE-analysis**) organizes the datasets and runnable codes used to reproduce the experiments and figures in the paper.

---

## Tutorials
The BRIDGE's neural network structure, pretraining protocol and inference protocol can be described as
![illustrate](/illustrate.png)

A quick demo for applying BRIDGE to a delayed transcription-splicing model is in [example](Tutorials.ipynb).


## Repository Structure

1. **run BRIDGE/** contains all inference codes used in the manuscript.
    - **inference_BRIDGE2d.jl**: BRIDGE inference code for a two-species delayed transcription-splicing model (Fig. 2).
    - **inference_BRIDGE3d.jl**: BRIDGE inference code for a three-species system including protein production and degradation (Fig. 3).
    - **inference_BRIDGE_Feedback.jl**: BRIDGE inference code for auto-regulatory
    feedback network (AFN) combined linear mapping approximation (LMA) (Fig. 4).
    - **Inference_BRIDGE_Toggle.jl**: BRIDGE inference code for a toggle-switch gene regulatory network combined linear mapping approximation (LMA) (Fig. 5).
    - **Inference_BRIDGE_capture_rate.jl**: BRIDGE inference code for a three-species system  including protein production and degradation with capture rate (Fig. 6).
    - **Inference_BRIDGE_ABC.jl**: Inference code for a two-species delayed transcription-splicing model using Approximate Bayesian Computation (ABC) algorithm (Fig. 2).
    - **Inference_BRIDGE_Exact.jl**: Inference code for a two-species delayed transcription-splicing model using exact solution (Fig. 2).
    - **Inference_BRIDGE_FSP.jl**: Inference code for a two-species delayed transcription-splicing model using maximum likelihood estimation with finite state projection (MLE+FSP) (Fig. 2).
    - **Inference_BRIDGE_MOM.jl**: Inference code for a two-species delayed transcription-splicing model using method of moments (MOM) (Fig. 2).
    - **Inference_BRIDGE_NNCME.jl**: Inference code for a two-species delayed transcription-splicing model using neural-network-aided chemical master equation method (NNCME) (Fig. 2).
2. **train_BRIDGE/** contains codes and data for training the BRIDGE in the manuscript.
      - **2d/** contains the BRIDGE training code and training dataset of two-species delayed transcription-splicing model.
      - **3d/** contains the BRIDGE training code and training dataset of three-species system including protein production and degradation.
3. **parameters_trained/** contains trained weights and bias of BRIDGE in the manuscript.
      - **params_trained2d.txt** trained weights and bias of BRIDGE for two-species delayed transcription-splicing model.
      - **params_trained3d.txt** trained weights and bias of BRIDGE for three-species system including protein production and degradation.
4. **dataset/** all datasets used in the manuscript
      - **synthetic_data/** contains the synthetic data for inference.
        - **counts_example2d.txt**: synthetic inference data for two-species delayed transcription-splicing model.
        - **counts_example3d.txt**: synthetic inference data for three-species system including protein production and degradation.
        - **counts_example_feedback.txt**: synthetic inference data for auto-regulatory
        feedback network (AFN).
        - **counts_example_toggle.txt**: synthetic inference data for toggle-switch gene regulatory network.
        - **counts_example_capture_rate.txt**: synthetic inference data three-species system  including protein production and degradation with capture rate
        - **β1β2.txt**: normalized capture rate β1β2.
        - **ps_forinfer_2d.txt**: kinectic parameters for Inference of two-species delayed transcription-splicing model.
        - **ps_forinfer_3d.txt**: kinectic parameters for Inference of three-species system including protein production and degradation.
    - **cluster/** contains the clustering code and cell annotations for real_data/
        - **cluster_ID/**: cell annotations for CTCL and ctrl dataset.
        - **cluster_CTCL.py**: clustering code for CTCL dataset.
        - **cluster_ctrl.py**: clustering code for ctrl dataset.
    - **real_data/**
        - **190426_CTCL.loom**: the raw CTCL `.loom` file containing both spliced and unspliced count matrices.
        - **190426_ctrl.loom**: the raw ctrl `.loom` file containing both spliced and unspliced count matrices.
        - **GSM3596101_CTCL-ADT-count.csv**: the raw CTCL `.csv` file containing protein count matrices.
        - **GSM3596096_ctrl-ADT-count.csv**: the raw ctrl `.csv` file containing protein count matrices.

---

## Corresponding package
The package can be installed through the Julia package manager:
```julia
] https://github.com/X-Y-Zhou/BRIDGE.jl
using BRIDGE
```



