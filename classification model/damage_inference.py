
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
from sys import exit
import cv2
import datetime

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict

import tensorflow as tf
import keras
from model import * 

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense
from model import *

# Configurations
NUM_WORKERS = 4
NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_EPOCHS = 120
LEARNING_RATE = 0.0001
RANDOM_SEED = 123
LOG_DIR = '/tmp/inference/classification_log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


damage_intensity_encoding = dict() 
damage_intensity_encoding[3] = 'destroyed' 
damage_intensity_encoding[2] = 'major-damage'
damage_intensity_encoding[1] = 'minor-damage'
damage_intensity_encoding[0] = 'no-damage'


###
# Creates data generator for validation set
###
def create_generator(test_df, test_dir, output_json_path):

    gen = keras.preprocessing.image.ImageDataGenerator(
                             rescale=1.4)

    try:
        gen_flow = gen.flow_from_dataframe(dataframe=test_df,
                                   directory=test_dir,
                                   x_col='uuid',
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   seed=RANDOM_SEED,
                                   class_mode=None,
                                   target_size=(128, 128))
    except:
        # No polys detected so write out a blank json
        blank = {}
        with open(output_json_path , 'w') as outfile:
            json.dump(blank, outfile)
        exit(0)


    return gen_flow

# Runs inference on given test data and pretrained model
def run_inference(test_data, test_csv, model_weights, output_json_path):

   model = generate_xBD_baseline_model()
   model.load_weights(model_weights)

   adam = keras.optimizers.Adam(lr=LEARNING_RATE,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=None,
                                    decay=0.0,
                                    amsgrad=False)


   model.compile(loss=ordinal_loss, optimizer=adam, metrics=['accuracy'])

   df = pd.read_csv(test_csv)

   test_gen = create_generator(df, test_data, output_json_path)
   test_gen.reset()
   samples = df["uuid"].count()

   steps = np.ceil(samples/BATCH_SIZE)

   tensorboard_callbacks = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

   predictions = model.predict_generator(generator=test_gen,
                    callbacks=[tensorboard_callbacks],
                    verbose=1)

   predicted_indices = np.argmax(predictions, axis=1)
   predictions_json = dict()
   for i in range(samples):
       filename_raw = test_gen.filenames[i]
       filename = filename_raw.split(".")[0]
       predictions_json[filename] = damage_intensity_encoding[predicted_indices[i]]

   with open(output_json_path , 'w') as outfile:
       json.dump(predictions_json, outfile)


def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--test_data',
                        required=True,
                        metavar="/path/to/xBD_test_dir",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--test_csv',
                        required=True,
                        metavar="/path/to/xBD_test_csv",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--model_weights',
                        default=None,
                        metavar='/path/to/input_model_weights',
                        help="Path to input weights")
    parser.add_argument('--output_json',
                        required=True,
                        metavar="/path/to/output_json")

    args = parser.parse_args()

    run_inference(args.test_data, args.test_csv, args.model_weights, args.output_json)


if __name__ == '__main__':
    main()
