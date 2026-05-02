# Bayesian Black-Box Optimization Capstone

This repository contains the ongoing work for the Bayesian Black-Box Optimization (BBO) capstone project. The general goal is to optimize eight unknown synthetic "black-box" functions which mirror real-world problems where underlying structural mechanisms are hidden or overly complicated, requiring targeted observations to maximize results.

By leveraging Surrogate Models (Gaussian Processes) and Acquisition Functions (Expected Improvement), we iteratively discover the maximal point for 8 separate functions ranging broadly dimensionally from 2D input spaces up to 8D.

## Structure and Reproducibility 
The codebase is centered around reproducibility and iterative data addition.

- `Bayesian_Optimization_Workflow.ipynb`: Core logic utilizing `scikit-learn` for our Bayesian Optimization.
- `data/`: Contains `.npy` arrays with our historical queries, though these are externally referenced as they are not committed to Git.
- `Datasheet.md` & `Model_Card.md`: Outlines assumptions and limitations regarding the data sets and mathematical models.