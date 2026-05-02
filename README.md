# Bayesian Black-Box Optimization Capstone — Stage 2

This repository serves as the documented record for **Stage 2** of the Bayesian Black-Box Optimization (BBO) capstone project. 

## Project Purpose & Objectives
The central objective of this project is to reliably find the maximum outputs for eight synthetic, unknown "black-box" functions. In real-world machine learning contexts, we often encounter scenarios—such as drug formulation testing or hyperparameter tuning—where evaluating a solution is incredibly slow or computationally expensive. By treating these 8 functions as opaque processes, this capstone simulates those real-world constraints. The objective is to efficiently locate the absolute peak performance for each mathematical system while conducting as few external queries as possible.

## Data Structure: Inputs and Outputs
The functions dynamically range in structural complexity, spanning from highly simple 2-dimensional boundaries up to incredibly complex 8-dimensional hyperspaces. 
- **Inputs**: Array vectors constrained identically between `[0.000000, 0.999999]` across $D$ dimensions.
- **Outputs**: A single floating-point objective scalar denoting the evaluated yield (which may naturally be negative or positive depending on internal penalties).
Data histories are accumulated locally as NumPy (`.npy`) tensors and updated iteratively upon processing new feedback.

## Technical Approach & Methodological Insights

### Theoretical Context: Regression, SVMs, and Bayesian Modeling
When attempting to map a complex space, traditional supervised models run into limits. Mapping highly curved, non-linear, and multi-modal boundaries severely violates the core assumptions of classical **Linear Regression** or threshold-based **Logistic Regression**. While more advanced approaches like **Support Vector Machines (SVMs)** utilizing custom kernels might successfully navigate non-linear spaces locally, they aren't inherently structured for uncertainty quantification—which is the single most essential requirement for weighing exploration versus exploitation inside severely data-starved setups.

Because we are restricted to only ~10-40 baseline evaluation points initially, the project pivots optimally toward formally structured **Bayesian Optimization**. By leveraging Surrogate Models (**Gaussian Processes** equipped with a non-linear Matern Kernel) we derive a robust probabilistic representation. The algorithm outputs not just a predicted mean, but a variance vector (uncertainty representation) allowing an Acquisition Function (**Expected Improvement**) to systematically evaluate whether to aggressively exploit known peaks or safely explore totally unknown valleys. 

### Iterative Process & Progress (Weeks 1-3+)
This capstone is fundamentally grounded in **iterative modelling**:
1. **Weeks 1 & 2 (Manual Heuristics)**: Because the initial baseline models were highly uncertain and unstable, early query choices relied heavily on structured, diversity-driven manual exploration. Weak signals on specific datasets (Functions 5 and 8) allowed for targeted perturbation-based exploitation, but higher dimensional distributions largely resisted unstructured guessing.
2. **Stage 2 / Week 3+ (Algorithmic Shift)**: Having manually accumulated enough iterative feedback to establish a foundational signal landscape across the 8 models, we subsequently shifted directly into the automated Gaussian Process architecture mapped in our repository. This allows the sophisticated mathematics to iteratively update the surrogate model upon each submission, making the search for the absolute maxima significantly more informed and statistically targeted.

## Structure and Reproducibility 
- `Bayesian_Optimization_Workflow.ipynb`: Core Python logic utilizing `scikit-learn` for our formalized Bayesian Optimization iterations.
- `execute_bbo.py`: Clean executable mirror to the notebook logic for pipeline automation.
- `data/`: Local directory housing `.npy` arrays containing historical query matrices (ignored by version control to avoid pushing raw data structures).
- `Datasheet.md` & `Model_Card.md`: Outlines assumptions and limitations regarding the datasets and surrogate boundaries.