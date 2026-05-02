# Datasheet

## Motivation
- **For what purpose was the dataset created?** It was generated as the basis for a curriculum-directed capstone project exploring hyperparameter tuning and model optimization without explicit knowledge of the function behavior.
- **Who created the dataset?** Course instructors / automated assessment platform.

## Composition
- **What do the instances that comprise the dataset represent?** The data represent synthetic test inputs alongside their corresponding function evaluation outputs. 
- **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** Only a fraction of the search space which ranges between `[0.0, 0.999999]^d`.
- **Is there any missing data?** No missing values; however, points are uniformly explored initially before being targeted by BO.

## Maintenance
- **Is the dataset updated?** Yes, iteratively across feedback submissions.
