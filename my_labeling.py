__authors__ = '1526641'
__group__ = 'TO_BE_FILLED'

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

def RetrievalByShape(img, shape): 
    
    knn = KNN(train_imgs, train_class_labels)
    
    knn.get_k_neighbours(img, 1)
    knnclass = knn.get_class()
    
    # select all the classes that matches the query
    img = img[knnclass == shape]
    
    return img

def Get_shape_accuracy(img, query, test):
    knn = KNN(train_imgs, train_class_labels)
    knn.get_k_neighbours(img, 2)
    knnclass = knn.get_class()
        
    valid = (knnclass == test) [knnclass == query]
    percent = np.count_nonzero(valid) / valid.shape[0] *100
        
    return percent

Get_shape_accuracy(test_imgs, 'X',test=test_class_labels)






