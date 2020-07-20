import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.python import InteractiveSession, ConfigProto
from PIL import Image, ImageFont, ImageDraw
import emoji

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def prepare_img(path):
    return cv2.resize(path, (256, 256)).reshape(1, 256, 256, 3)


current_dir = os.path.dirname(os.path.realpath(__file__))
model = tf.keras.models.load_model(os.path.join(current_dir, 'model', 'model.h5'))

rock_img = cv2.imread(os.path.join(current_dir, 'emoji', 'rock.png'))
paper_img = cv2.imread(os.path.join(current_dir, 'emoji', 'paper.png'))
scissors_img = cv2.imread(os.path.join(current_dir, 'emoji', 'scissors.png'))

print('Loaded model from disk')

shape_to_label = {'rock': np.array([0., 1., 0.]), 'paper': np.array([1., 0., 0.]), 'scissors': np.array([0., 0., 1.])}
arr_to_shape = {np.argmax(shape_to_label[x]): x for x in shape_to_label.keys()}
label_to_img = {'rock': rock_img, 'paper': paper_img, 'scissors': scissors_img}

cap = cv2.VideoCapture(0)


def draw_overlay(src, prediction_name):
    emoji_img = label_to_img[prediction_name]
    emoji_rows, emoji_cols, _ = emoji_img.shape
    frame_rows, frame_cols, _ = src.shape

    overlay = cv2.addWeighted(src[0:emoji_rows, frame_cols-emoji_cols:frame_cols], 0.5, emoji_img, 1.0, 0)

    src[0:emoji_rows, frame_cols-emoji_cols:frame_cols] = overlay


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

    height, width, channels = gray.shape

    resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    raw_prediction = model.predict(prepare_img(gray[0:width, 0:height]))
    print(raw_prediction)
    argmax = np.argmax(raw_prediction)
    print(argmax)
    label = arr_to_shape[argmax]
    print(label)
    draw_overlay(gray, label)

    cv2.imshow('Dataset', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
