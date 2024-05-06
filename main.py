#%%
# Import libraries
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np


#%%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


#%%
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize mean, var, and priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
#%%
# Defining data for the dataframe
current_dir = os.getcwd()

data_path = os.path.join(current_dir, 'spambase.csv')

data = pd.read_csv(data_path)

#%%
# do a 80 20 split
training = data[:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(data):
    training = data.iloc[train_index]
    test = data.iloc[test_index]
