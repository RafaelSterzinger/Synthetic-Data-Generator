import argparse
import os
import numpy as np
import ml3.config as cfg

from PIL import Image
from ml3.gan.generator import load_generator
from tensorflow.keras.models import Model


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


def run(dataset: str, epoch: int):
    dir = f'data/fake/{dataset}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = f'data/splits/{dataset}/train'
    for _class in os.listdir(path):

        _class_path = f'{dir}/{_class}'
        if not os.path.exists(_class_path):
            os.mkdir(_class_path)

        generator = load_generator(f'models/{dataset}/{_class}', epoch)
        generate_images(generator, _class_path, 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True, help='name of dataset')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='specification of the model')
    args = parser.parse_args()

    run(args.dir, args.epochs)
