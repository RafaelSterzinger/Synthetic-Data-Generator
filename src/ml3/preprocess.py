import argparse
import os

import split_folders
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

import ml3.config as cfg


def __split_data(folder: str):
    split_folders.ratio('data/original/' + folder, 'data/splits/' + folder, seed=1337,
                        ratio=(cfg.TRAIN, cfg.VALIDATION))


def train_data_generator(folder: str, augmentation=False) -> DirectoryIterator:
    if augmentation:
        train_data_generator = ImageDataGenerator(rescale=cfg.SCALE, horizontal_flip=True, brightness_range=[0.2, 1.0],
                                                  zoom_range=[0.5, 1.0], rotation_range=25)
    else:
        train_data_generator = ImageDataGenerator(rescale=cfg.SCALE)
    generator = train_data_generator.flow_from_directory(
        folder,
        target_size=(cfg.SIZE, cfg.SIZE),
        batch_size=cfg.EVAL_BATCH_SIZE,
        class_mode='categorical',
        interpolation='lanczos')
    return generator


def validation_data_generator(folder: str) -> DirectoryIterator:
    validation_data_generator = ImageDataGenerator(rescale=cfg.SCALE)
    generator = validation_data_generator.flow_from_directory(
        'data/splits/' + folder + '/val',
        target_size=(cfg.SIZE, cfg.SIZE),
        class_mode='categorical',
        interpolation='lanczos')
    return generator


def run(dataset: str):
    __split_data(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True, help='name of folder')
    args = parser.parse_args()

    run(args.dir)
