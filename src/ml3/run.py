import argparse

import ml3.config as cfg
import os

# https://pathmind.com/wiki/generative-adversarial-network-gan
from ml3.gan.gan import GAN
from ml3.preprocess import train_data_generator
import ml3.eval.eval as eval


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
    print(f"Training GAN, path {path}, class {class_name}")
    gan = GAN()
    gan.train(data_generator)
    return gan.generator


def generate_images(gan, fake_path):
    print(f'Generating fake images in {fake_path}')
    # gan.generate()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True, help='name of folder to train the model on',
                        default='data')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=cfg.EVAL_EPOCHS)
    parser.add_argument('-m', '--mode', choices=['real', 'augm', 'fake'], required=True, help='mode for training',
                        default='real')
    args = parser.parse_args()
    if args.mode == 'fake':
        fake_path = 'data/fake/' + args.dir
        path = f'data/splits/{args.dir}/train'
        for dir in os.listdir(path):
            generator = train_gan(args.dir, dir)
            generate_images(generator, fake_path + '/' + dir)
    eval.run(args.dir, args.epochs, args.mode)
