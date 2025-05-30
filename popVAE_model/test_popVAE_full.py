
import os
import random
import numpy as np
import tensorflow as tf
import gc

from sklearn.model_selection import train_test_split

import popVAE_full as popVAE
import data
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

def main():
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    technique="vae"

    weight_b6,weight_b7,weight_b8=0.7,0.13,0.17
    patch_size=11
    bands=10
    bands_context=bands
    epoch=100
    batch_size= 64 # training 64 
    latent_dim=900
    choice='EPKD_12_Bands'
    input_raster = f'{choice}.tif'
    output_raster = f'{choice}_processed.tif'
    output_prediction=f'{choice}_popVAE_full_{latent_dim}_predicted.tif'
    plots_path=f'plots_{choice}'
    ins_population_csv = 'ins_population.csv'
    target_raster_path = "POP.tif"
    
    predicted_output_raster = f'{choice}_predicted_{weight_b6}_{weight_b7}_{weight_b8}.tif'
    input_data = data.preprocess_raster_compososite(input_raster, output_raster,weight_b6,weight_b7,weight_b8)

    district_masks = data.load_district_masks()

    for i, mask in enumerate(district_masks):
        height, width = mask.shape
        print(f"Width: {width}, Height: {height}")
        break
    
    ins_population = data.load_ins_population_data(ins_population_csv)
    target_data, profile = data.load_and_preprocess_target_data(target_raster_path)
    gc.collect()


    if(technique=="vae"):
        attention=0
        kernel_attention=5
        reduction_ratio=16
        
        # Create the model
        model = popVAE.create_vae_pipeline_model_simple(patch_size, latent_dim, bands, bands_context,attention,reduction_ratio,kernel_attention)
        # Get the total number of trainable parameters
        trainable_params = get_trainable_params(model)
        ratio_dataset=(height * width) / (trainable_params * 10)
        print(f"Total number of trainable parameters in the model: {trainable_params}")

        print (f"ratio_dataset=(height * width) / (trainable_params * 10)=({height} * {width}) / ({trainable_params} * 10)={ratio_dataset}")

        model_json = model.to_json()
        with open(f"model_architecture_popVAE_full_latendim_{latent_dim}.json", 'w') as json_file:
            json_file.write(model_json)
            
        all_indices , height, width = popVAE.get_all_indices(ratio_dataset, input_data, patch_size,bands)
        input_data_reshaped = input_data.reshape((height, width, bands))
        # Split the indices into train, val, test
        train_indices, temp_indices = train_test_split(all_indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        # Initialize data generators

        training_generator = popVAE.DataGenerator(train_indices, input_data_reshaped, target_data, patch_size, bands, bands_context, batch_size)
        validation_generator = popVAE.DataGenerator(val_indices, input_data_reshaped, target_data, patch_size, bands, bands_context, batch_size)
        test_generator = popVAE.DataGenerator(test_indices, input_data_reshaped, target_data, patch_size, bands, bands_context, batch_size)

        # Load the best weights if they exist
        try:
            model.load_weights(f"best_weights_popVAE_full_latendim_{latent_dim}.h5") 
        except:
            print("No best weights found. Starting training from scratch.")
        
        checkpoint = ModelCheckpoint(f"best_weights_popVAE_full_latendim_{latent_dim}.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
        
        # Train the model using the data generators with callbacks
        history = model.fit(training_generator, 
                             validation_data=validation_generator, 
                             epochs=epoch, 
                             callbacks=[checkpoint, early_stop])        
        #Evaluate the model on the test set
        test_loss = model.evaluate(test_generator)
        print(f'Test Loss: {test_loss}')
        model.save(f"best_weights_popVAE_full_latendim_{latent_dim}.h5" )
        
       
        predicted_image = popVAE.predict_and_reconstruct(model, input_data, profile, output_prediction, patch_size, bands, bands, batch_size_p=5000, 
            checkpoint_file=f"checkpoint_popVAE_full_ltent_dim_{latent_dim}.txt")


    print("Evaluating Start")
    # ev.calculate_pixel_r2(target_data, predicted_image)


    ev.calculate_district_r2(predicted_image, district_masks, ins_population)
    print ("latent_dim=",latent_dim)
    ev.save_plots(history, plots_path)
    print("Evaluating Finish")

        
if __name__ == "__main__":
    main()

