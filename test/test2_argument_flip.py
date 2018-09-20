import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
import glob
import os
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np


PATH='D:/BaiduYunDownload/faces/right'
SAVE_PATH='D:/BaiduYunDownload/faces/right_flip'

for image_name in os.listdir(PATH):
    #for imgstr in os.listdir(os.path.join(PATH,folder)):
    imgstr = os.path.join(PATH, image_name)
    img = Image.open(os.path.join(PATH + image_name, imgstr)).transpose(Image.FLIP_LEFT_RIGHT)
    img = img.resize((100,180))
    img.save(os.path.join(SAVE_PATH,'trans_'+image_name))
# y_train = keras.utils.to_categorical(y_train, 7)
