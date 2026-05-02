# Bayesian Black-Box Optimization Capstone

This repository contains the ongoing work for the Bayesian Black-Box Optimization (BBO) capstone project. The general goal is to optimize eight unknown synthetic "black-box" functions which mirror real-world problems where underlying structural mechanisms are hidden or overly complicated, requiring targeted observations to maximize results.

## Project Methodology & Reflection

### Weeks 1 & 2: Manual Exploration & Heuristics
Initially, with only 10-40 starting data points per function, the mathematical models were found to be highly uncertain. The approach for Week 1 focused heavily on **diversity-driven exploration**, manually targeting unrepresented regions scattered across the `[0, 1]` domain while avoiding extreme boundaries. The higher-dimensional functions proved to be the most challenging to conceptually query without explicitly formal signals given the massive sparsity of the space.

In Week 2, the approach transitioned to a **mixed exploration-exploitation strategy**. Functions that previous returned strong outputs (e.g., Functions 5 and 8) were targeted for early exploitation using controlled value perturbations, while poorer performing functions (e.g., Functions 1, 3, 4, and 6) were allocated continued exploration to escape non-productive local valleys. 

### Week 3+: Formalizing Bayesian Optimization
Because the true output boundaries are highly curved, non-linear, and multi-modal, simpler linear regression approximations violate spatial assumptions. Having accumulated a critical baseline of manual probing data, the project logically hands the reins to automated Bayesian Optimization. 

By aggressively leveraging Surrogate Models (**Gaussian Processes** with a Matern Kernel) and Acquisition Functions (**Expected Improvement**), we can intuitively and mathematically track the maximal points for all 8 functions far more reliably than manual exploration across the more complex 2D-8D spaces.

## Structure and Reproducibility 
The codebase is centered around reproducibility and iterative data addition.

- `Bayesian_Optimization_Workflow.ipynb`: Core logic utilizing `scikit-learn` for our formalized Bayesian Optimization.
- `data/`: Contains `.npy` arrays with our historical queries, though these are externally referenced as they are not committed to Git.
- `Datasheet.md` & `Model_Card.md`: Outlines assumptions and limitations regarding the data sets and mathematical models.