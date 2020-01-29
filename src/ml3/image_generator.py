import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model

import ml3.config as cfg
from ml3.gan.generator import load_generator

def generate_images(generator: Model, path: str, count):
    noise = np.random.normal(0, 1, (count, cfg.SEED_SIZE))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    upper_limit = np.vectorize(lambda x: 1 if x > 1 else x)
    under_limit = np.vectorize(lambda x: 0 if x < 0 else x)

    gen_imgs = upper_limit(gen_imgs)
    gen_imgs = under_limit(gen_imgs)
    gen_imgs *= 255.0

    for index in range(count):
        im = Image.fromarray(gen_imgs[index].astype(np.uint8))
        im.save(f"{path}/{index}.jpg")


def run(dataset: str):
    directory = f'data/fake/{dataset}'
    if not os.path.exists(directory):
        os.mkdir(directory)

    path = f'data/splits/{dataset}/train'
    for _class in os.listdir(path):

        _class_path = f'{directory}/{_class}'
        if not os.path.exists(_class_path):
            os.mkdir(_class_path)

        generator = load_generator(f'models/{dataset}/{_class}', cfg.EPOCH_OF_MODEL)
        generate_images(generator, _class_path, cfg.IMAGE_AMOUNT)
