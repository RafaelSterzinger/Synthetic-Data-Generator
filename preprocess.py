import os
from tensorflow.keras.preprocessing.image import load_img, save_img
import PIL
import split_folders

# %% load data
path = 'data'
data = {}
for root, dirs, files in os.walk(path):
    label = root.split('/')[-1]
    images = []
    for name in files:
        if name.endswith('.jpg'):
            image_path = root + '/' + name
            img = load_img(image_path)
            images.append(img)
    data[label] = images

# %% resize images
print(data)
data["apples"][0].resize((224, 224), PIL.Image.LANCZOS).show()

# %% split data

split_folders.ratio('data/original', 'data', seed=1337, ratio=(.8, .1, .1))
