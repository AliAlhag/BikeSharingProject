1. bike_random_forest.py

This script:

    - Downloads and extracts the London Bike Sharing Dataset using the Kaggle API.
    - Preprocesses the data by handling cyclical features, scaling numeric data, and encoding categorical data.
    - Implements a Random Forest Regressor model with optimal hyperparameters (n_estimators=100, max_depth=None).
    - Evaluates the model using Mean Squared Error (MSE) and R^2.

2. bike_sharing_mlp.py

This script:

    - Performs the same data download, preprocessing, and feature engineering as bike_random_forest.py.
    - Implements a Multi-Layer Perceptron (MLP) Regressor model with optimal hyperparameters (hidden_layer_sizes=(100,), activation='relu', max_iter=10000).
    - Evaluates the MLP model using MSE and R^2.

3. bike_decision_tree.py

This script:

    - Mirrors the data handling and preprocessing steps from the other two files.
    - Implements a Decision Tree Regressor model with optimal hyperparameters (max_depth=10).
    - Evaluates the Decision Tree model using MSE and R^2.


## Model Evaluation Results

**Decision Tree** - MSE: 103438.68722215774, R²: 0.9136746974221632

**Random Forest** - MSE: 62722.59853659777, R²: 0.9476545242157701

**MLP Regressor** - MSE: 67578.53900099674, R²: 0.9436019734618254
