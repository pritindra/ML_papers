"""
You only look once : paper implementation

"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

train_dir = 'train_zip/train'
test_dir = 'test_zip/test'

images = [cv2.imread(image, cv2.IMREAD_UNCHANGED) for image in sorted(os.listdir(train_dir)) if image[-4:]=='.jpg']

print(images[0].shape)

