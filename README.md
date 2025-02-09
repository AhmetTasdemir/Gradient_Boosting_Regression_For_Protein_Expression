Note: The study is in the process of being published in the European Biophysics Journal, so I cannot share the dataset yet.

# Gradient Boosting Regression for Protein Expression Prediction

## Overview

This notebook demonstrates the use of a Gradient Boosting Regressor (GBR) to predict protein expression levels. It includes data preprocessing, hyperparameter tuning using GridSearchCV, model evaluation, and performance metrics.

## Dependencies

Ensure you have the following Python libraries installed before running the notebook:

pip install pandas scikit-learn

## Dataset

The dataset dataset.csv is loaded into a Pandas DataFrame.

The column variables is dropped as it is not needed for modeling.

protein_expression is set as the target variable (y).

The remaining columns are used as features (X).

## Preprocessing Steps

Train-Test Split: The dataset is split into training (80%) and testing (20%) sets.

Feature Scaling: RobustScaler is used to normalize the feature set.

## Model Training

A GradientBoostingRegressor is initialized and trained.

Hyperparameter tuning is performed using GridSearchCV with 5-fold cross-validation.

The best set of hyperparameters is identified and used to fit the final model.

## Hyperparameter Tuning

The following parameters are optimized:

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0]
}

## Model Evaluation

The model's performance is evaluated using:

Mean Squared Error (MSE)

R² Score

The best hyperparameters, MSE, and R² score are printed.

## Output

Best Gradient Boosting Regressor parameters.

MSE and R² score of the final model.

## Usage

To run the notebook:

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

Then execute the cells sequentially to preprocess the data, train the model, and evaluate its performance.

Author

Created by Ahmet Taşdemir

