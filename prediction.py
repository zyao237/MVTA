import numpy as np
import pandas as pd
import math
from collections import OrderedDict
import sys
from scipy import spatial
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial import distance
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
# from fastFM import als
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle

class Evaluator:
    def __init__(self, data_dic):
        self.x_test = data_dic['x_test']
        self.y_test = data_dic['y_test']
        self.x_train = data_dic['x_train']
        self.y_train = data_dic['y_train']

        # self.x_test = pd.read_csv(dir + 'x_test_'+model_name+'.csv', skipinitialspace=True,
        #                                  header=None).as_matrix()
        # self.x_train = pd.read_csv(dir + 'x_train_'+model_name+'.csv', skipinitialspace=True,
        #                                  header=None).as_matrix()
        # self.y_test = pd.read_csv(dir + 'y_test_'+model_name+'.csv', skipinitialspace=True,
        #                                  header=None).as_matrix().flatten()
        # self.y_train = pd.read_csv(dir + 'y_train_'+model_name+'.csv', skipinitialspace=True,
        #                                  header=None).as_matrix().flatten()
    def linear_regression(self):
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(self.x_train, self.y_train)
        # Make predictions using the testing set
        y_pred = regr.predict(self.x_test)
        # The mean squared error
        print("Mean squared error: %.6f"
              % mean_squared_error(self.y_test, y_pred))
        print("Mean absolute error: %.6f"
              % mean_absolute_error(self.y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.6f' % r2_score(self.y_test, y_pred))

    def multi_layer_Perceptron(self):
        mlp = MLPRegressor(hidden_layer_sizes=(256, 128),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=5000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)
        mlp.fit(self.x_train, self.y_train)
        y_pred = mlp.predict(self.x_test)
        # The mean squared error
        print("Mean squared error: %.6f"
              % mean_squared_error(self.y_test, y_pred))
        print("Mean absolute error: %.6f"
              % mean_absolute_error(self.y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.6f' % r2_score(self.y_test, y_pred))

    def k_nearest_neighbor(self, K):
        n_neighbors = K
        weight = 'distance'
        knn = KNeighborsRegressor(n_neighbors, weights=weight)
        knn.fit(self.x_train, self.y_train)
        y_pred = knn.predict(self.x_test)
        # The mean squared error
        print("Mean squared error: %.6f"
              % mean_squared_error(self.y_test, y_pred))
        print("Mean absolute error: %.6f"
              % mean_absolute_error(self.y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        print('R-squared score: %.6f' % r2_score(self.y_test, y_pred))
        print('Explained variance score: %.6f' % explained_variance_score(self.y_test, y_pred))

    def gbdt(self):
        est = GradientBoostingRegressor(n_estimators=1000, max_depth=3, learning_rate=0.1)
        est.fit(self.x_train, self.y_train)
        y_pred = est.predict(self.x_test)
        # The mean squared error
        print("Mean squared error: %.6f"
              % mean_squared_error(self.y_test, y_pred))
        print("Mean absolute error: %.6f"
              % mean_absolute_error(self.y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.6f' % r2_score(self.y_test, y_pred))





