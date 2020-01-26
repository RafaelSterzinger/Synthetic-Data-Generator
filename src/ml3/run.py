import numpy as np
import ml3.config as cfg
import matplotlib.pyplot as plt
from ml3.eval import *
from ml3.gan import generator, discriminator
from PIL import Image
import tensorflow as tf

# https://pathmind.com/wiki/generative-adversarial-network-gan
from ml3.gan.gan import GAN
from ml3.config import *
from ml3.preprocess import train_data_generator

def train(epoch: int, batch_size=128):
    pass

    # generator = Generator.generator()

    # for epoch in range(epoch):
    #     # create fake images
    #     noise = np.random.normal(0, 1, (1,cfg.latent_dim))
    #     fake_img = generator.predict(noise)
    #     img = Image.fromarray(np.squeeze(fake_img))
    #     img.show()

    # train discriminator
    # dis_loss_real = discriminator.predict_on_batch()


def train_gan(path, class_name):
    data_generator = train_data_generator(path, classes=[class_name])
    # print(data_generator)
    # print(data_generator.class_indices)

    # train_dataset = tf.data.Dataset.from_tensor_slices(data_generator).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    gan = GAN()
    gan.train(data_generator)


if __name__ == '__main__':
    train_gan('fruit', 'apples')
