# Hyperparamter_optimization
Hyperparameter Optimization using Hyperband and Bayesian Optimization on bike sharing data

This repository demonstrates hyperparameter optimization (HPO) techniques on the bike_sharing dataset using mlr3verse in R. It includes:

Decision Tree (rpart) tuning with Hyperband
XGBoost tuning with Bayesian Optimization (MBO) vs. Random Search
Benchmarking of different tuning strategies

## 1. Data & Task Setup
Load the bike_sharing dataset from mlr3data.
Inspect data structure (task$data(), autoplot()).

## 2. Decision Tree (rpart) with Hyperband
Pipeline: Robustify data → subsample → train rpart.
Search Space: Tunes cp, minisplit, and subsampling fraction.
Tuner: Hyperband (aggressive early-stopping, eta = 3).

## 3. XGBoost with Bayesian Optimization (MBO)
Surrogate Model: Random Forest (regr.ranger).
Acquisition Function: Expected Improvement (EI).
Optimizer: Random search for acquisition function optimization.

## 4. Benchmarking
Compares:
Untuned XGBoost
XGBoost + Random Search
XGBoost + Bayesian Optimization
Baseline (regr.featureless)

## 🎯 Key Methods
## 🔹 Hyperparameter Optimization
Hyperband: Resource-efficient (adapts subsample.frac as budget).
Bayesian Optimization (MBO):
Loop Function: bayesopt_ego
Surrogate: Gaussian Process / Random Forest.
Acquisition: Expected Improvement (EI).

## 🔹 Model Pipelines
Robustify: Handles missing values, factors, and scaling.
Subsampling: Used as a budget parameter for Hyperband.


# 📜 License
MIT License.