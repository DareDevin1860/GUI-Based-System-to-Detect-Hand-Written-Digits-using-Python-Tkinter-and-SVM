from sklearn import datasets
digits=datasets.load_digits()
data=digits.data
target=digits.target
images=digits.images

print(len(target))
print(data.shape)

print(data[100])

from matplotlib import pyplot as plt


import cv2

image=images[1]
image=cv2.resize,(200,200)

cv2.imshow(images[1])
plt.show()
