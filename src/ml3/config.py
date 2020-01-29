# General
SIZE = 64  # Do not change this value (64)
IMAGE_SHAPE = (SIZE, SIZE, 3)

# Pre processing
TRAIN = 0.8
VALIDATION = 0.2
SCALE = 1 / 255

# Data augmentation
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = [0.2, 1.0]
ZOOM_RANGE = [0.5, 1.0]
ROTATION_RANGE = 25

# GAN training
GAN_EPOCHS = 10
GAN_BATCH_SIZE = 12  # Batch size must be smaller than amount of samples
SAVE_INTERVAL = 1
SAVE_IMAGES = True
LEARNING_RATE_DISCRIMINATOR = 0.0002
LEARNING_RATE_GENERATOR = 0.0002
SEED_SIZE = 100  # Random noise size to generate images

# Image generation
IMAGE_AMOUNT = 5
EPOCH_OF_MODEL = GAN_EPOCHS - 1  # Here you can specify which generation of the model should generate the fake images

# Data evaluation
EVAL_EPOCHS = 4
EVAL_BATCH_SIZE = 128
