import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import ml3.config as cfg
from ml3.gan.discriminator import build_discriminator
from ml3.gan.generator import build_generator


class GAN():
    def __init__(self, path: str, _class: str):
        self.__create_dirs(_class, path)
        self.__create_model()

    def __create_model(self):
        d_optimizer = Adam(lr=cfg.LEARNING_RATE_DISCRIMINATOR, beta_1=0.5, beta_2=0.999)
        g_optimizer = Adam(lr=cfg.LEARNING_RATE_GENERATOR, beta_1=0.5, beta_2=0.999)
        # discriminator
        self.discriminator = build_discriminator()
        self.discriminator.compile(
            loss="binary_crossentropy", optimizer=d_optimizer, metrics=["accuracy"])

        # generator
        self.generator = build_generator()

        # model
        self.combined_model = self.__build_combined()
        self.combined_model.compile(
            loss="binary_crossentropy", optimizer=g_optimizer)

    def __create_dirs(self, _class, path):
        dir = f"models/{path}/{_class}"
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.models = dir

        dir = f"images/{path}/{_class}"
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.images = dir

    def __build_combined(self):
        # set trainable=False to only train the generator depending on the accuracy of the discriminator
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])

        return model

    def train(self, epochs, trainings_data, batch_size, save_interval):
        half_batch = int(batch_size / 2)
        num_batches = int(trainings_data.shape[0] / half_batch)
        print("Number of Batches : ", num_batches)

        history = []
        prev_g_loss = 0
        prev_d_loss = 0

        for epoch in range(epochs):
            for iteration in range(num_batches):
                # generator create fake images
                noise = np.random.normal(0, 1, (half_batch, cfg.SEED_SIZE))
                fake_imgs = self.generator.predict(noise)

                # sample real images
                idx = np.random.randint(0, trainings_data.shape[0], half_batch)
                real_imgs = trainings_data[idx]

                # train discriminator, with real_imgs = 1 and fake_imgs = 0
                d_loss_real = self.discriminator.train_on_batch(
                    real_imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(
                    fake_imgs, np.zeros((half_batch, 1)))
                # average of fake and real loss
                d_loss = np.add(d_loss_real, d_loss_fake) / 2

                # train generator, with discriminator predicting 1
                noise = np.random.normal(0, 1, (batch_size, cfg.SEED_SIZE))
                valid_y = np.array([1] * batch_size)
                g_loss = self.combined_model.train_on_batch(noise, valid_y)

                # smoothing loss and appending to the history
                prev_d_loss = d_loss[0] * 0.05 + prev_d_loss * 0.95
                prev_g_loss = g_loss * 0.05 + prev_g_loss * 0.95
                history.append([prev_d_loss, prev_g_loss, d_loss[1]])

                print("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    epoch, iteration, prev_d_loss, 100 * d_loss[1], prev_g_loss))

            if cfg.SAVE_IMAGES:
                self.__save_imgs(epoch)

            if epoch % save_interval == 0:
                self.__save_model(epoch)

        return history

    def __save_model(self, epoch):
        self.generator.save(f'{self.models}/model_epoch_{epoch}.h5')

    def __save_imgs(self, epoch):
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


# plot validation loss, history = [dis_loss, gen_loss, dis_acc]
def __plot_loss_combine(history_gan: []):
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
    plt.xlabel("Training iterations", fontsize=15)
    plt.ylabel("Training loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


def run(dataset: str):
    dir = f'models/{dataset}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir = f'images/{dataset}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir = f'plots/{dataset}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    # create a generator for each class of dataset
    for _class in os.listdir(f'data/splits/{dataset}/train'):
        print(f'Start class {_class}')
        real_imgs = []
        img_list = glob.glob(f'data/splits/{dataset}/train/{_class}/*')
        for img_path in img_list:
            img = img_to_array(load_img(img_path, target_size=(cfg.SIZE, cfg.SIZE), interpolation='lanczos'))
            real_imgs.append(img)

        # normalize image
        real_imgs = np.asarray(real_imgs)
        real_imgs = (real_imgs.astype(np.float32) - 127.5) / 127.5

        gan = GAN(dataset, _class)
        history = gan.train(epochs=cfg.GAN_EPOCHS, trainings_data=real_imgs, batch_size=cfg.GAN_BATCH_SIZE,
                            save_interval=cfg.SAVE_INTERVAL)

        __plot_loss_combine(history)
        plt.savefig(dir + f"/{_class}_loss.png")
