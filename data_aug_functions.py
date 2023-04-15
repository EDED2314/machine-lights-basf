from typing import List
from numpy import expand_dims
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator
import os
import cv2
from matplotlib import pyplot

def dataAugGeneral(img_names: List[str], mode: str):
    for img_name in img_names:
        path = os.path.join("data", "resized", img_name)
        img = load_img(path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        try:
            os.mkdir(os.path.join("data", mode))
        except Exception as e:
            pass
            
            
        if mode == "V":
            datagen = ImageDataGenerator(height_shift_range=0.2)
            
        elif mode == "H":
            datagen = ImageDataGenerator(width_shift_range=[-50,50])
        elif mode == "B":
            datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        else:
            # mode == "rotate"
            datagen = ImageDataGenerator(rotation_range=90)
            
        it = datagen.flow(samples, batch_size=1)
        for i in range(4):
            # define subplot
            # pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            name = img_name.replace(".jpg", "")
            
            im_rgb = cv2.cvtColor( batch[0].astype('uint8'), cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(os.path.join("data", mode, f"{name}_{i}.jpg"),im_rgb)
            # print(type( batch[0].astype('uint8')))
        #     image = batch[0].astype('uint8')
        #     # plot raw pixel data
        #     pyplot.imshow(image)
                    
        # pyplot.show()

def allData():
    for name in ["V", "H", "B", "rotate"]:
        dir_names = os.listdir(os.path.join("data", name))
        
        for dirname in dir_names:
            length = len(os.listdir(os.path.join("data", "processed")))
            img = cv2.imread(os.path.join("data", name, dirname))
            cv2.imwrite(os.path.join("data", "processed", f"{length+1}.jpg"), img)

            