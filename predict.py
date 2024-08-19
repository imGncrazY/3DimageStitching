from Model.src.models.mnist_module import MNISTLitModule
from lightning import Trainer
import numpy as np
import torch
import os
import cv2

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_show(predictions):
    predictions = np.transpose(predictions, [1, 2, 3, 0])
    for prediction in predictions[:]:
        cv_show('img', prediction)

dm = 'your/checkpoints/path'
model = MNISTLitModule()
trainer = Trainer(accelerator="gpu", devices=2)
predictions = trainer.predict(model, dm)
img_show(predictions)