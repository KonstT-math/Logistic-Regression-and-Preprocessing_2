# Logistic-Regression-and-Preprocessing_2
Logistic regression in pure python and in pytorch, preprocessing of dataset breast-cancer.csv

Remarks:

the dataset 'breast-cancer.csv' has no missing values

we follow the preprocessing steps below:

1) we need to standardize the dataset - rescale with mean 0 and std 1 (rescale.py)

2) we drop the 'id' variable, since it has no context in our model

3) categorical variable 'diagnosis' needs numerical values 0 and 1 (for 'M' and 'B')

4) target variable 'diagnosis' is moved at the end of the dataframe in order to apply our logistic regression

(steps 2,3,4 via preprocess0.py)


Contents:

csv files:

breast-cancer.csv : the dataset is publically available on the Kaggle website (has 30 features and a target variable with 2 categories)

Preprocessing files:
preprocess0.py, rescale.py

Logistic regression in pure python:
lreg.py, LR_train.py

Logistic regression in pytorch:
logReg_torch.py
