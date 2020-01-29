# General
SIZE = 64  # Do not change this value (64)
IMAGE_SHAPE = (SIZE, SIZE, 3)

# Pre-processing
TRAIN = 0.8
VALIDATION = 0.2
SCALE = 1 / 255

# GAN
GAN_EPOCHS = 10
GAN_BATCH_SIZE = 12         # batch size must be smaller than amount of samples
SAVE_INTERVAL = 1
LEARNING_RATE_DISCRIMINATOR = 0.0002
LEARNING_RATE_GENERATOR = 0.0002

# Image generation
IMAGE_AMOUNT = 5
EPOCH_OF_MODEL = GAN_EPOCHS - 1  # Here you can specify which generation of the model should generate the fake images

# Evaluation
EVAL_EPOCHS = 4
EVAL_BATCH_SIZE = 128

# Size of random noise
SEED_SIZE = 100
