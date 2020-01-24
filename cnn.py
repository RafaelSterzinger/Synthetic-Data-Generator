import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# %% load data
batch_size = 128
# All images will be rescaled by 1./255
scale = 1.0 / 255
size = (200, 200)
train_data_generator = ImageDataGenerator(rescale=scale)
validation_data_generator = ImageDataGenerator(rescale=scale)
test_data_generator = ImageDataGenerator(rescale=scale)
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_data_generator.flow_from_directory(
    'data/train',  # This is the source directory for training images
    target_size=size,  # All images will be resized to 200 x 200
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode='categorical')
validation_generator = validation_data_generator.flow_from_directory(
    'data/val',  # This is the source directory for training images
    target_size=size,  # All images will be resized to 200 x 200
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode='categorical')
test_generator = test_data_generator.flow_from_directory(
    'data/test',  # This is the source directory for training images
    target_size=size,  # All images will be resized to 200 x 200
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode='categorical')

class_amount = len(set(train_generator.classes))

# %% build cnn model

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(class_amount, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

# %% train stuff

n_epochs = 30
total_sample = train_generator.n
history = model.fit(
    train_generator,
    steps_per_epoch=int(total_sample / batch_size),
    epochs=n_epochs,
    verbose=1,
    validation_data=validation_generator)

# %% plot training

plt.figure(figsize=(7, 4))
plt.plot([i + 1 for i in range(n_epochs)], history.history['acc'], '-o', c='k', lw=2, markersize=9)
plt.grid(True)
plt.title("Training accuracy with epochs\n", fontsize=18)
plt.xlabel("Training epochs", fontsize=15)
plt.ylabel("Training accuracy", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# %% plot loss

plt.figure(figsize=(7, 4))
plt.plot([i + 1 for i in range(n_epochs)], history.history['loss'], '-o', c='k', lw=2, markersize=9)
plt.grid(True)
plt.title("Training loss with epochs\n", fontsize=18)
plt.xlabel("Training epochs", fontsize=15)
plt.ylabel("Training loss", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
