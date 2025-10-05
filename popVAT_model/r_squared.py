import os
import random
import numpy as np
import tensorflow as tf
import gc
import argparse

from sklearn.model_selection import train_test_split

import CNNPxL
import data_atrous as data
import evaluate as ev

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import rasterio


def get_trainable_params(model):
    return int(np.sum([K.count_params(w) for w in model.trainable_weights]))

def set_random_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args):
    
    set_random_seeds(42)

    ins_population = data.load_ins_population_data(args.csv)
    district_masks = data.load_district_masks(args.region, args.nb_masks)
    target_data, profile = data.load_and_preprocess_target_data_without_log(args.raster)
    gc.collect()

    print("Evaluating Start")
    ev.calculate_district_r2(target_data, district_masks, ins_population)
    print("Evaluating Finish of:", args.raster)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted population raster")
    parser.add_argument('--raster', type=str, required=True, default="tunisia10_popVAE_full_GAG_1024_30_best_weights_popVAE_full_Tunisia_GAG_1024_30.h5_predicted.tif", help="Path to the predicted raster (.tif)")
    parser.add_argument('--csv', type=str, required=True,  default="ins_population.csv", help="Path to INSEE population CSV file")
    parser.add_argument('--region', type=str, required=True,  default="Tunisia",  help="Region name (e.g., France)")
    parser.add_argument('--nbmasks', type=int, required=True, default="24", help="Number of district masks to load")

    args = parser.parse_args()
    main(args)

"""
python3 r_squared.py --raster tunisia10_popVAE_full_GAG_1024_30_best_weights_popVAE_full_Tunisia_GAG_1024_30.h5_predicted.tif --csv ins_population.csv --region Tunisia --nb_masks 24
"""