import argparse
import os

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

import ml3.config as cfg
from ml3.eval.plot import *
from ml3.preprocess import train_data_generator, validation_data_generator


def build_cnn_model(class_amount: int):
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
        # The first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(cfg.SIZE, cfg.SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a dense layer
        tf.keras.layers.Flatten(),
        # 128 neuron in the fully-connected layer
        tf.keras.layers.Dense(128, activation='relu'),
        # 5 output neurons for 5 classes with the softmax activation
        tf.keras.layers.Dense(class_amount, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])

    return model


def train(path: str, episodes: int):
    train_generator = train_data_generator(path)
    validation_generator = validation_data_generator(path)

    class_amount = len(set(train_generator.classes))
    total_sample = train_generator.n

    model = build_cnn_model(class_amount)
    history = model.fit(
        train_generator,
        steps_per_epoch=int(total_sample / cfg.BATCH_SIZE),
        epochs=episodes,
        verbose=1,
        validation_data=validation_generator)

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True, help='name of folder to train the model on')
    parser.add_argument('-e', '--episodes', type=int, help='number of episodes', default=cfg.EPISODES)
    parser.add_argument('-m', '--mode', choices=['real', 'fake'], required=True, help='mode for training',
                        default='real')
    args = parser.parse_args()

    model, history = train(args.dir, args.episodes)

    plot_loss(history)
    plot_training(history)
    plot_validation(history)
    plot_validation_loss(history)

    path = 'models/' + args.dir + '/' + args.mode + '.h5'

    try:
        # Create target Directory
        os.mkdir('models/' + args.dir)
        print("Created directory")
    except FileExistsError:
        print("Directory already exists")

    model.save(path, overwrite=True)
