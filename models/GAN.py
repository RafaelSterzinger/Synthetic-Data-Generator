import numpy as np
import config as cfg
from models import Discriminator, Generator
import matplotlib.pyplot as plt


def train(epoch: int, batch_size=128):
    generator = Generator.generator()
    discriminator = Discriminator.discriminator()

    for epoch in range(epoch):
        # create fake images
        noise = np.random.normal(cfg.feature_dim)
        fake_img = generator.predict(noise)
        plt.imshow(fake_img)

        # train discriminator
        dis_loss_real = discriminator.predict_on_batch()




if __name__ == '__main__':
    train(100)
