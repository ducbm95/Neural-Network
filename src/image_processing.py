import cv2
import mnist_loader
import network
import cPickle
import numpy as np
 
def resize_image(in_file):
    im = cv2.imread(in_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
     
    find_top = find_left = False
    top = left = 1000
    bot =right = 0
     
    height, width = im.shape
    for i in range(0, height):
        for j in range(0, width):
            if im[i][j] < 50:
                if not find_top and i < top:
                    top = i
                    find_top = True
                if i > bot:
                    bot = i
                if not find_left and j < left:
                    left = j
                    find_left = True
                if j > right:
                    right = j
        find_left = False
    
    padding = 30
    crop_im = im[top - padding:bot + padding, left - padding:right + padding]
 
    resized_image = cv2.resize(crop_im, (28, 28)) 
    return resized_image

def normalize_image(img):
    def func(x):
        return (255.0 - x) / 255.0

    out = np.vectorize(func)(img)
    return out

def train_network():
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    f = open("network.pickle", "wb")
    cPickle.dump(net, f)
    file.close()
    
def detect_image(img):
    f = open("network.pickle", "rb")
    net = cPickle.load(f)
    f.close()
    
    img = np.reshape(img, (784, 1))
    out = np.argmax(net.feedforward(img))
    return out

# train_network()