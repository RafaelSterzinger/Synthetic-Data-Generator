import os

import ml3.config as cfg
from ml3.eval.plot import *
from ml3.preprocess import train_data_generator, validation_data_generator
from ml3.eval.cnn import build_cnn
from tensorflow.keras.callbacks import History
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def train(dataset: str, epochs: int, mode: str) -> History:
    if mode == 'fake':
        path = 'data/fake/' + dataset
    else:
        path = 'data/splits/' + dataset + '/train'

    train_generator = train_data_generator(path, augmentation=True if mode == 'augm' else False)
    validation_generator = validation_data_generator(dataset)

    class_amount = len(set(train_generator.classes))
    total_sample = train_generator.n

    model = build_cnn(class_amount)

    history = model.fit(
        train_generator,
        steps_per_epoch=int(total_sample / cfg.EVAL_BATCH_SIZE),
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator)

    print_report(model, validation_generator)

    return history


def print_report(model, validation_generator):
    Y_pred = model.predict_generator(validation_generator, validation_generator.n // cfg.EVAL_BATCH_SIZE + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    target_names = [key for key in validation_generator.class_indices]
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


def run(dataset: str, mode: str) -> History:
    plot_path = f'plots/{dataset}'
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    history = train(dataset, cfg.EVAL_EPOCHS, mode)

    plot_loss(history)
    plt.savefig(plot_path + f'/{mode}_loss.png')
    plot_training(history)
    plt.savefig(plot_path + f'/{mode}_acc.png')
    plot_validation(history)
    plt.savefig(plot_path + f'/{mode}_val_acc.png')
    plot_validation_loss(history)
    plt.savefig(plot_path + f'/{mode}_val_loss.png')

    return history
