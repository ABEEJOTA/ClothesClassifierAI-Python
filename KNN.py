__authors__ = ['1526641']
__group__ = 'DJ.15'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.train_data = np.array(train_data.reshape([train_data.shape[0], -1]), dtype = 'float')
        #Shape[0] devuelve el numero de filas de la matriz
        #Reshape canviamos el numero de filas x -1

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        size = test_data.shape
        # test_data = test_data.reshape(size[0], size[1] * size[2])
        test_data = test_data.reshape(size[0], np.prod(size[1:]))
        dist = cdist(test_data, self.train_data)
        e = np.argsort(dist, axis=1)[:, :k]
        self.neighbors = self.labels[e]


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        most_voted = []
        percentage_vots = []

        for i in self.neighbors:
            arr, index, freq = np.unique(i, return_index=True, return_counts=True)
            arr = arr[np.argsort(index)]
            freq = freq[np.argsort(index)]

            most_voted.append(arr[np.argmax(freq)])
            percentage_vots.append((max(freq) / i.size) * 100)



        return most_voted, percentage_vots


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        