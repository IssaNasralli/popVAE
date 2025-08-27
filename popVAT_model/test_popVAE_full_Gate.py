
import os
import random
import numpy as np
import tensorflow as tf
import gc
import argparse

from sklearn.model_selection import train_test_split
import popVAE_full_Gate_Atrous_Gate as popVAE

import data_atrous as data
import evaluate as ev
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import rasterio


def set_files(choice, country, model_option, batch_size, latent_dim, patch_size_global, weights):
    input_raster = f'{choice}.tif'
    output_raster = f'{choice}_processed.tif'
    ins_population_csv = 'ins_population.csv'
    if (country=="France"):
        ins_population_csv = 'insee_population.csv'
        
    target_raster_path = f"POP_{country}.tif"
    if weights is None:
        weights=f"best_weights_popVAE_full_{country}_{model_option}_{batch_size}_{latent_dim}_{patch_size_global}.h5"
    output_prediction=f'{choice}_popVAE_full_{model_option}_{batch_size}_{latent_dim}__{weights}__{patch_size_global}_predicted.tif'
    json=f"model_architecture_popVAE_full_{country}_{model_option}_{batch_size}_{latent_dim}_{patch_size_global}.json"
    checkpoint_file=f"checkpoint_popVAE_full_{country}_{model_option}_{batch_size}_{latent_dim}_{patch_size_global}_{weights}.txt"
    return input_raster, output_raster, ins_population_csv, target_raster_path, output_prediction, weights, json, checkpoint_file


def set_param(parser):

    parser.add_argument("--model_option", type=str, required=True, help="Model option for ablation study") # G (VAE + Gate_Z), GA (VAE + Gate_Z + Atrous) and GAG (VAE + Gate_Z + Atrous + Gate_A)
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training/evaluation")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension size")
    parser.add_argument("--patch_size_global", type=int, required=True, help="Global patch size")
    parser.add_argument("--training", type=int, choices=[0, 1], required=True, help="Training mode: 1 for training, 0 for inference")
    parser.add_argument("--choice", type=str, required=True, help="Input raster base name (e.g., tunisia10)")
    parser.add_argument("--country", type=str, required=True, help="Country name for district mask loading")
    parser.add_argument("--nb_masks", type=int, required=True, help="Number of district masks to load")
    parser.add_argument("--weights", type=str, required=False, default=None, help="File Weights for transfer learning (optional)")

    args = parser.parse_args()
    model_option = args.model_option
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    patch_size_global = args.patch_size_global
    training = args.training
    choice = args.choice
    country = args.country
    nb_masks = args.nb_masks
    weights = args.weights

    print("model_option =", model_option)
    print("training =", training)
    print("batch_size =", batch_size)
    print("latent_dim =", latent_dim)
    print("patch_size_global =", patch_size_global)
    print("choice =", choice)
    print("country =", country)
    print("nb_masks =", nb_masks)
    print("weights =", weights)
    return model_option, batch_size, latent_dim, patch_size_global, training, choice, country, nb_masks, weights

def get_trainable_params(model):
    return int(np.sum([K.count_params(w) for w in model.trainable_weights]))

def set_random_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    parser = argparse.ArgumentParser(description="popVAT training and inference pipeline")
    model_option, batch_size, latent_dim, patch_size_global, training, choice, country, nb_masks, weights = set_param(parser)
    set_random_seeds(42)

    weight_b6,weight_b7,weight_b8=0.7,0.13,0.17
    patch_size=11
    bands=10
    bands_context=bands
    epoch=100
    input_raster, output_raster, ins_population_csv, target_raster_path, output_prediction, weights, json, checkpoint_file =set_files(choice, country, model_option, batch_size, latent_dim, patch_size_global, weights)

    input_data, profile = data.preprocess_raster_compososite(input_raster, output_raster,weight_b6,weight_b7,weight_b8, bands+2)


    district_masks = data.load_district_masks(country, nb_masks)

    for i, mask in enumerate(district_masks):
        height, width = mask.shape
        print(f"Width: {width}, Height: {height}")
        break
    
    # Create the model
    if(model_option=="G"):
        model = popVAE.create_vae_G(patch_size, latent_dim, bands, bands_context)
    else:
        if(model_option=="GA"):
            model = popVAE.create_vae_GA(patch_size, patch_size_global, latent_dim, bands, bands_context)
        else:
            model = popVAE.create_vae_GAG(patch_size, patch_size_global, latent_dim, bands, bands_context)

    # Load the best weights if they exist
    noweights=0
    try:
        model.load_weights(weights) 
    except:
        print("No best weights found. Starting training from scratch.")
        noweights=1
        
    gc.collect()

    checkpoint = ModelCheckpoint(weights, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
   
   # Train the model using the data generators with callbacks
    if(training==1):
        # Get the total number of trainable parameters
        trainable_params = get_trainable_params(model)
        ratio_dataset=(height * width) / (trainable_params * bands)
        print(f"Total number of trainable parameters in the model: {trainable_params}")
        print (f"ratio_dataset={ratio_dataset}")
        model_json = model.to_json()
        with open(json, 'w') as json_file:
            json_file.write(model_json)
        if(model_option!="G"):    
            all_indices , height, width = popVAE.get_all_indices(ratio_dataset, input_data, patch_size_global,bands)
        else:
            all_indices , height, width = popVAE.get_all_indices(ratio_dataset, input_data, patch_size,	bands)
            
        input_data_reshaped = input_data.reshape((height, width, bands))
        # Split the indices into train, val, test
        train_indices, temp_indices = train_test_split(all_indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        # Initialize data generators
        target_data, profile = data.load_and_preprocess_target_data(target_raster_path)
        if(model_option!="G"):
            training_generator = popVAE.DataGeneratorAtrous(train_indices, input_data_reshaped, target_data, patch_size, patch_size_global, bands, bands_context, batch_size)
            validation_generator = popVAE.DataGeneratorAtrous(val_indices, input_data_reshaped, target_data, patch_size, patch_size_global, bands, bands_context, batch_size)
            test_generator = popVAE.DataGeneratorAtrous(test_indices, input_data_reshaped, target_data, patch_size, patch_size_global, bands, bands_context, batch_size)
        else:
            training_generator = popVAE.DataGenerator(train_indices, input_data_reshaped, target_data, patch_size, bands, bands_context, batch_size)
            validation_generator = popVAE.DataGenerator(val_indices, input_data_reshaped, target_data, patch_size, bands, bands_context, batch_size)
            test_generator = popVAE.DataGenerator(test_indices, input_data_reshaped, target_data, patch_size, bands, bands_context, batch_size)
    
        history = model.fit(training_generator, 
                             validation_data=validation_generator, 
                             epochs=epoch, 
                             callbacks=[checkpoint, early_stop])        
        #Evaluate the model on the test set
        test_loss = model.evaluate(test_generator)
        print(f'Test Loss: {test_loss}')
        model.save(weights)
        noweights=0
   
    if(noweights==0):
        profile.update(dtype=rasterio.float32, count=1, nodata=0.0)
        # Create the model
        if(model_option=="G"):
            predicted_image = popVAE.predict_and_reconstruct_G(model, input_data, profile, output_prediction, patch_size, bands, bands, 5000, checkpoint_file=checkpoint_file)
        else:
            predicted_image = popVAE.predict_and_reconstruct_GAG_GA(model, input_data, profile, output_prediction, patch_size, patch_size_global, bands, bands, 5000, checkpoint_file=checkpoint_file)
            
        print("Evaluating Start")

        ins_population = data.load_ins_population_data(ins_population_csv)
        ev.calculate_district_r2(predicted_image, district_masks, ins_population)
        print("Evaluating Finish of:")
        print ("model_option = ", model_option)
        print ("Country = ", country)
        print ("training = ", training)
        print ("batch_size = ", batch_size)
        print ("latent_dim = ", latent_dim)
        print ("patch_size_global = ", patch_size_global)  
        print ("weights = ", weights)  
    else:
        print("No weights for prediction.")

if __name__ == "__main__":
    main()

	
	#python3 test_popVAE_full_Gate.py --model_option 2 --batch_size 64 --latent_dim 900 --patch_size_global 35 --training 1 --choice tunisia10 --country Tunisia --nb_masks 24 --weights weights.h5
