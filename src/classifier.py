import cv2
import numpy as np
import network

from sklearn import datasets

class Classifier():
    
    def __init__(self):
        """MNIST dataset include 70000 28x28 image"""
        mnist = datasets.fetch_mldata("MNIST Original")
        target = [self.vectorized_result(j) for j in mnist.target]
        hog_data = [self.cal_hog(x) for x in mnist.data]
        
        self.net = network.Network([324, 30, 10])
        self.net.SGD(zip(hog_data[0:60000], target[0:60000]), 30, 10, 3.0, test_data=zip(hog_data[60000:70000], mnist.target[60000:70000]))
    
    """
    input is array 784 element of image 28x28
    """
    def predict(self, image):
        hog_img = self.cal_hog(image)
        out = np.argmax(self.net.feedforward(hog_img))
        return out
    
    """
    input is array 784 element of image 28x28
    """
    def cal_hog(self, image):
        # initialize HOG
        winSize = (28, 28)
        blockSize = (14, 14)
        blockStride = (7, 7)
        cellSize = (7, 7)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hog_img = hog.compute(np.reshape(image, (28, 28)))
        return hog_img
    
    def vectorized_result(self, j):
        e = np.zeros((10, 1))
        e[int(j)] = 1.0
        return e
        
