
from PIL import Image
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import random
import argparse
import logging
import json
import cv2
import datetime

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict

import tensorflow as tf
import keras
import ast
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Add, Input, Concatenate
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras import backend as K


###
# Loss function for ordinal loss from https://github.com/JHart96/keras_ordinal_categorical_crossentropy
###
def ordinal_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * keras.losses.categorical_crossentropy(y_true, y_pred )


###
# Generate a simple CNN
###
def generate_xBD_baseline_model():
  weights = 'imagenet'
  inputs = Input(shape=(128, 128, 3))

  base_model = ResNet50(include_top=False, weights=weights, input_shape=(128, 128, 3))

  for layer in base_model.layers:
    layer.trainable = False

  x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(128, 128, 3))(inputs)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

  x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

  x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

  x = Flatten()(x)

  base_resnet = base_model(inputs)
  base_resnet = Flatten()(base_resnet)

  concated_layers = Concatenate()([x, base_resnet])

  concated_layers = Dense(2024, activation='relu')(concated_layers)
  concated_layers = Dense(524, activation='relu')(concated_layers)
  concated_layers = Dense(124, activation='relu')(concated_layers)
  output = Dense(4, activation='relu')(concated_layers)

  model = Model(inputs=inputs, outputs=output)
  return model
