# pre processing
TRAIN = 0.8
VALIDATION = 0.2
SCALE = 1 / 255

# general
SIZE = 128
IMAGE_SHAPE = (SIZE, SIZE)

# evaluation
EVAL_EPOCHS = 30
EVAL_BATCH_SIZE = 128

# Generation resolution - Must be square
# Training data is also scaled to this.
# Note GENERATE_RES 4 or higher  will blow Google CoLab's memory and have not
# been tested extensivly.
GENERATE_RES = 4  # Generation resolution factor (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES  # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
DATA_PATH = 'data'
EPOCHS = 128
BUFFER_SIZE = 60000
