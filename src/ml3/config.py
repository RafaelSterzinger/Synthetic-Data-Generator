# General
SIZE = 64  # Do not change this value
IMAGE_SHAPE = (SIZE, SIZE, 3)

# Pre-processing
TRAIN = 0.8
VALIDATION = 0.2
SCALE = 1 / 255

# GAN
GAN_EPOCHS = 10
GAN_BATCH_SIZE = 12
SAVE_INTERVAL = 1

# Image generation
IMAGE_AMOUNT = 5
EPOCH_OF_MODEL = GAN_EPOCHS - 1  # Here you can specify which generation of the model should generate the fake images

# Evaluation
EVAL_EPOCHS = 4
EVAL_BATCH_SIZE = 128

# Size of random noise
SEED_SIZE = 100
