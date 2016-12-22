import cv2
import datetime as dt

start = dt.datetime.now()
from skimage.feature import hog
end = dt.datetime.now()
print (end - start)

hog_ = cv2.HOGDescriptor()
im = cv2.imread("temp.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
im = cv2.resize(im, (64, 128))

# cv2.imshow("aa", im)
# cv2.waitKey()

start = dt.datetime.now()
h = hog_.compute(im)
end = dt.datetime.now()
print h.shape
print (end - start)

print im.shape


start = dt.datetime.now()
h1 = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
end = dt.datetime.now()
print (end - start)
print h1.shape
print h1
# print (end - start)