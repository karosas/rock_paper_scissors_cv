import tensorflow as tf
import logging
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python import InteractiveSession, ConfigProto
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Flatten, Dense, MaxPool2D
import matplotlib
matplotlib.use('TkAgg')

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tqdm.tqdm = tqdm.auto.tqdm


batch_size = 8
epochs = 20
IMG_HEIGHT = 256
IMG_WIDTH = 256


def train():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    train_dir = os.path.join(current_dir, 'train')
    validation_dir = os.path.join(current_dir, 'validation')

    train_paper_dir = os.path.join(train_dir, 'paper')
    train_rock_dir = os.path.join(train_dir, 'rock')
    train_scissors_dir = os.path.join(train_dir, 'scissors')

    validation_paper_dir = os.path.join(validation_dir, 'paper')
    validation_rock_dir = os.path.join(validation_dir, 'rock')
    validation_scissors_dir = os.path.join(validation_dir, 'scissors')

    num_paper_tr = len(os.listdir(train_paper_dir))
    num_rock_tr = len(os.listdir(train_rock_dir))
    num_scissors_tr = len(os.listdir(train_scissors_dir))

    num_paper_val = len(os.listdir(validation_paper_dir))
    num_rock_val = len(os.listdir(validation_rock_dir))
    num_scissors_val = len(os.listdir(validation_scissors_dir))

    total_train = num_paper_tr + num_rock_tr + num_scissors_tr
    total_val = num_paper_val + num_rock_val + num_scissors_val

    print('total train:', total_train)
    print('total validation:', total_val)

    train_image_generator = ImageDataGenerator(rescale=1./255)#, horizontal_flip=True, rotation_range=45, zoom_range=0.5)
    validation_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=45,
                                                    zoom_range=0.5)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  shuffle=True,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='categorical')

    densenet = DenseNet121(include_top=False, weights='imagenet', classes=3, input_shape=(256, 256, 3))
    densenet.trainable = True
    model = Sequential()
    model.add(densenet)
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_data_gen,
        steps_per_epoch=math.ceil(total_train / batch_size),
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=math.ceil(total_val / batch_size)
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model_path = os.path.join(current_dir, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save(os.path.join(model_path, 'model.h5'))


if __name__ == '__main__':
    train()
