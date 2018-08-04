import cv2
import numpy as np

im_pth = 'trainannot/0006R0_f01260.png'

im = cv2.imread(im_pth)
im0 = im[..., 0]
im1 = im[..., 1]
print(im.shape)
print(im0.max())
print(im0.min())

px = set()

print(np.sum(im0 == im1))
print(im0.shape)


im0 *= 20
im1 *= 20
cv2.imshow('0', im0)
cv2.imshow('1', im1)
cv2.waitKey(0)
