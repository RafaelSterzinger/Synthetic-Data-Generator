import argparse

from ml3 import preprocess, image_generator
from ml3.eval import eval
from ml3.gan import gan
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt


def plot_comparision(real: History, augm: History, fake: History, mode: str, title: str, ylab: str):
    plt.figure(figsize=(8, 5))
    plt.plot([i + 1 for i in range(len(real.epoch))], real.history[mode], '-', c='green', lw=2, markersize=9)
    plt.plot([i + 1 for i in range(len(augm.epoch))], augm.history[mode], '-', c='red', lw=2, markersize=9)
    plt.plot([i + 1 for i in range(len(fake.epoch))], fake.history[mode], '-', c='blue', lw=2, markersize=9)
    plt.grid(True)
    plt.legend(['Original', 'Augmentation', 'Fake'])
    plt.title(f"{title}\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel(ylab, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


if __name__ == '__main__':
    # choose dataset ("Edit Configurations" -> Parameters: -d <dataset-name>
    #                                       -> Working Directory: <project root>)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, help='name of the dataset')
    args = parser.parse_args()
    dataset = args.dataset

    count = 1
    print(f'{count}: Starting pre-processing')
    preprocess.run(dataset)

    count += 1
    print(f'{count}: Starting training GAN\'s')
    gan.run(dataset)

    count += 1
    print(f'{count}: Starting with image generation')
    image_generator.run(args.dataset)

    count += 1
    print(f'{count}: Starting evaluation of original')
    real = eval.run(dataset, mode="real")

    count += 1
    print(f'{count}: Starting evaluation of augmentation')
    augm = eval.run(dataset, mode="augm")

    count += 1
    print(f'{count}: Starting evaluation of fake')
    fake = eval.run(dataset, mode="fake")

    plot_comparision(real, augm, fake, 'acc', 'Comparision of the training accuracy', 'Training accuracy')
    plt.savefig(f"plots/{dataset}/comparision_acc.png")
    plot_comparision(real, augm, fake, 'loss', 'Comparision of the training loss', 'Training loss')
    plt.savefig(f"plots/{dataset}/comparision_loss.png")
    plot_comparision(real, augm, fake, 'val_acc', 'Comparision of the validation accuracy', 'Validation accuracy')
    plt.savefig(f"plots/{dataset}/comparision_val_acc.png")
    plot_comparision(real, augm, fake, 'val_loss', 'Comparision of the validation loss', 'Validation loss')
    plt.savefig(f"plots/{dataset}/comparision_val_loss.png")
