import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
import glob
import os
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np


def preprocess_input(x):
    #print(x)
    #print(x.shape)
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# 设置生成器参数
datagen = image.ImageDataGenerator(featurewise_center=False,
                                   featurewise_std_normalization=False,
                                   zca_whitening=False,
                                   rotation_range=15.,
                                   shear_range=0.05,
                                   zoom_range=0.05,
                                   height_shift_range=0.05,
                                   width_shift_range=0.05,
                                   #rescale=1./255,
                                   horizontal_flip=False,
                                   preprocessing_function=preprocess_input)

PATH='D:\BaiduYunDownload\MergeTest\\'
SAVE_PATH='D:\BaiduYunDownload\SaveTest\\'

x_all = np.array([]).reshape((-1, 224, 224, 1))
for folder in os.listdir(PATH):
    for imgstr in os.listdir(os.path.join(PATH,folder)):
        img = load_img(os.path.join(PATH+folder,imgstr),grayscale=True,target_size=(224,224))  # this is a PIL image
        x_all = np.append(x_all,img_to_array(img).reshape(1,224,224,1),axis=0)
# y_train = keras.utils.to_categorical(y_train, 7)

#datagen.fit(x_all)
gen_data = datagen.flow(x_all,batch_size=1,shuffle=False,save_to_dir=SAVE_PATH,save_prefix='gen')


# 生成9张图
for i in range(21):
    gen_data.next()
# 找到本地生成图，把9张图打印到同一张figure上
name_list = glob.glob(SAVE_PATH+'\*')
fig = plt.figure()
for i in range(21):
    img = Image.open(name_list[i])
    sub_img = fig.add_subplot(3,3,i%9+1)
    sub_img.imshow(img)
plt.show()