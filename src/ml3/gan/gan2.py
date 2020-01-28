import argparse
import glob
import os

import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

import ml3.config as cfg
import matplotlib.pyplot as plt
from ml3.gan.discriminator import build_discriminator
from ml3.gan.generator import build_generator


class GAN():
    def __init__(self, path: str, _class: str):
        self._create_dirs(_class, path)
        self._create_model()

    def _create_model(self):
        d_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        g_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        # discriminator
        self.discriminator = build_discriminator()
        self.discriminator.compile(
            loss="binary_crossentropy", optimizer=d_optimizer, metrics=["accuracy"])
        # generator
        self.generator = build_generator()
        # model
        self.combined_model = self.build_combined()
        self.combined_model.compile(
            loss="binary_crossentropy", optimizer=g_optimizer)

    def _create_dirs(self, _class, path):
        dir = f"models/{path}/{_class}"
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.models = dir

        dir = f"images/{path}/{_class}"
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.images = dir

    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        model.summary()

        return model

    def train(self, epochs, X_train, batch_size=128, save_interval=10):

        half_batch = int(batch_size / 2)
        num_batches = int(X_train.shape[0] / half_batch)
        print("Number of Batches : ", num_batches)
        history = []
        d_loss = []
        for epoch in range(epochs):
            for iteration in range(num_batches):
                # train discriminator
                if iteration % 1 == 0:
                    # generator create fake images
                    noise = np.random.normal(0, 1, (half_batch, cfg.SEED_SIZE))
                    gen_imgs = self.generator.predict(noise)

                    idx = np.random.randint(0, X_train.shape[0], half_batch)
                    imgs = X_train[idx]

                    # training
                    d_loss_real = self.discriminator.train_on_batch(
                        imgs, np.ones((half_batch, 1)))
                    d_loss_fake = self.discriminator.train_on_batch(
                        gen_imgs, np.zeros((half_batch, 1)))

                    # average of fake and real loss
                    d_loss = np.add(d_loss_real, d_loss_fake) / 2

                # train generator
                noise = np.random.normal(0, 1, (batch_size, cfg.SEED_SIZE))
                valid_y = np.array([1] * batch_size)

                g_loss = self.combined_model.train_on_batch(noise, valid_y)
                history.append([d_loss[0], g_loss, d_loss[1]])
                print("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    epoch, iteration, d_loss[0], 100 * d_loss[1], g_loss))

            self.save_imgs(epoch)

            if epoch % save_interval == 0:
                self.save_model(epoch)

        return history

    def save_model(self, epoch):
        self.combined_model.save_weights(
            f'{self.models}/model_epoch_{epoch}.h5')

    def load_model(self, epoch):
        self.combined_model.load_weights(
            f'{self.models}/model_epoch_{epoch}.h5')

    def save_imgs(self, epoch):
        r, c = 4, 4

        noise = np.random.normal(0, 1, (r * c, cfg.SEED_SIZE))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        upper_limit = np.vectorize(lambda x: 1 if x > 1 else x)
        under_limit = np.vectorize(lambda x: 0 if x < 0 else x)

        gen_imgs = upper_limit(gen_imgs)
        gen_imgs = under_limit(gen_imgs)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"{self.images}/epoch_{epoch}.png")

        plt.close()

    def generate_images(self, path: str, count):
        noise = np.random.normal(0, 1, (count, cfg.SEED_SIZE))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        upper_limit = np.vectorize(lambda x: 1 if x > 1 else x)
        under_limit = np.vectorize(lambda x: 0 if x < 0 else x)

        gen_imgs = upper_limit(gen_imgs)
        gen_imgs = under_limit(gen_imgs)
        gen_imgs *= 255.0

        # fig, axs = plt.subplots()
        for index in range(count):
            im = Image.fromarray(gen_imgs[index].astype(np.uint8))
            im.save(f"{path}/{index}.jpg")
            # axs.imshow(gen_imgs[index])
            # axs.axis('off')
            # fig.savefig(f"{path}/{index}.jpg")
            # index += 1


# %% plot validation loss
# history = [dis_loss, gen_loss, dis_acc]
def plot_loss_combine(history_gan: [], epochs: int):
    plt.figure(figsize=(8, 5))
    plt.plot([x[0] for x in history_gan], '-', lw=2, markersize=9,
             color='blue')
    plt.plot([x[1] for x in history_gan], '-', lw=2, markersize=9,
             color='orange')
    plt.plot([x[2] for x in history_gan], '--', lw=2, markersize=9,
             color='black')
    plt.grid(True)
    plt.legend(['Discriminator loss', 'Generator loss', 'Discriminator accuracy'])
    plt.title("Validation loss with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


def run(path: str, epochs: int, save_interval: int):
    dir = f'models/{path}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir = f'images/{path}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir = f'plots/{path}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    for _class in os.listdir(f'data/splits/{path}/train'):
        print(f'Start class {_class}')
        X_train = []
        img_list = glob.glob(f'data/splits/{path}/train/{_class}/*')
        for img_path in img_list:
            img = img_to_array(load_img(img_path, target_size=(cfg.SIZE, cfg.SIZE), interpolation='lanczos'))
            X_train.append(img)

        X_train = np.asarray(X_train)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        gan = GAN(path, _class)
        history = gan.train(epochs=epochs, X_train=X_train, batch_size=64, save_interval=save_interval)

        plot_loss_combine(history, epochs)
        plt.savefig(dir + f"/{_class}_loss.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True, help='name of folder')
    parser.add_argument('-e', '--epochs', type=int, default=cfg.GAN_EPOCHS, help='amount of epochs to train')
    parser.add_argument('-s', '--save_interval', type=int, default=cfg.SAVE_INTERVAL)
    args = parser.parse_args()

    run(args.dir, args.epochs, args.save_interval)
