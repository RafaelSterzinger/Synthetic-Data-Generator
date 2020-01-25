import argparse
import os
import split_folders
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import ml3.config as cfg


def split_data(folder: str):
    split_folders.ratio('data/original/' + folder, 'data/splits/' + folder, seed=1337,
                        ratio=(cfg.TRAIN, cfg.VALIDATION))


def train_data_generator(folder: str):
    train_data_generator = ImageDataGenerator(rescale=cfg.SCALE)
    generator = train_data_generator.flow_from_directory(
        'data/splits/' + folder + '/train',
        target_size=cfg.IMAGE_SHAPE,
        batch_size=cfg.BATCH_SIZE,
        class_mode='categorical')
    return generator


def validation_data_generator(folder: str):
    validation_data_generator = ImageDataGenerator(rescale=cfg.SCALE)
    generator = validation_data_generator.flow_from_directory(
        'data/splits/' + folder + '/val',
        target_size=cfg.IMAGE_SHAPE,
        class_mode='categorical')
    return generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True, help='name of folder')
    args = parser.parse_args()

    try:
        # Create target directory
        os.mkdir('data/splits')
        print("Created directory")
    except FileExistsError:
        print("Directory already exists")

    split_data(args.dir)