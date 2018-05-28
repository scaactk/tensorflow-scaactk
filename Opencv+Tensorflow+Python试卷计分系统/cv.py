import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
img = cv2.imread("D:\\test\\5.jpg")
res = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
#cv2.namedWindow("Image")
#print(img.shape)
#灰度化
emptyImage3=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
cv2.imshow("a",emptyImage3)
cv2.waitKey(0)
#二值化
ret, bin = cv2.threshold(emptyImage3, 140, 255,cv2.THRESH_BINARY)
cv2.imshow("a",bin)
cv2.waitKey(0)
print(bin)
def normalizepic(pic):
    im_arr = pic
    im_nparr = []
    for x in im_arr:
        x=1-x/255
        im_nparr.append(x)
    im_nparr = np.array([im_nparr])
    return im_nparr
#print(normalizepic(bin))
img=normalizepic(bin).reshape((1,784))
#print(img)
img= img.astype(np.float32)