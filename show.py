import numpy as np
import os
import cv2
import torch
import torchvision

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_show(predictions):
    predictions = np.transpose(predictions, [1, 2, 3, 0])
    for prediction in predictions[:]:
        cv_show('img', prediction)

root = 'E:\\PythonPrograms\\3D Imgs Stitching\\myproject\\data\\Mydata'
data_list = os.listdir(root)
for dirpath, dirnames, filenames in os.walk(root):
    for file in filenames:
        path = os.path.join(dirpath, file)
        data = np.load(path)
        data = data[1, :]
        img_show(data)
    