import os
import random
import numpy as np
import tensorflow as tf
import gc

from sklearn.model_selection import train_test_split

import popVAE_full_Gate_Atrous_Gate_CORAL as popVAE
import data_atrous as data
import evaluate as ev
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, Input
import rasterio
import itertools
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import Huber

import time

def get_trainable_params(model):
    return int(np.sum([K.count_params(w) for w in model.trainable_weights]))

def set_random_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_total_loss(xc_src, vae_output, z_mean_src, z_log_var_src, y_true, p_pred, coral_src, coral_tgt, 
                       weight_reconstruction=1.0, weight_kl=1.0, weight_prediction=1.0, weight_coral=1.0):
    # Reconstruction loss
    # reconstruction_loss = K.mean(MeanSquaredError()(xc_src, vae_output))
    # KL divergence loss    
    kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var_src - K.square(z_mean_src) - K.exp(z_log_var_src), axis=-1))

    # Prediction loss
    # prediction_loss = K.mean(MeanSquaredError()(y_true, p_pred))
    
    
    prediction_loss = K.mean(Huber()(y_true, p_pred))
    reconstruction_loss = K.mean(Huber()(xc_src, vae_output))

    # CORAL loss
    d = tf.cast(tf.shape(coral_src)[1], tf.float32)
    source_coral = compute_covariance(coral_src)
    target_coral = compute_covariance(coral_tgt)
    coral_loss = tf.reduce_sum(tf.square(source_coral - target_coral)) / (4 * (d ** 2))
    # print ("-------------------")
    # print (f"reconstruction_loss:{reconstruction_loss}")
    # print (f"kl_loss:{kl_loss}")
    # print (f"prediction_loss:{prediction_loss}")
    # print (f"coral_loss:{coral_loss}")
    # Weighted total loss
    total_loss = (weight_reconstruction * reconstruction_loss +
                  weight_kl * kl_loss +
                  weight_prediction * prediction_loss +
                  weight_coral * coral_loss)
    
    return total_loss

def compute_covariance(features):
    n = tf.cast(tf.shape(features)[0], tf.float32)
    mean = tf.reduce_mean(features, axis=0, keepdims=True)
    features_centered = features - mean
    cov = tf.matmul(features_centered, features_centered, transpose_a=True) / (n - 1)
    return cov

#@profile
#@tf.function
def train_step(x_source, x_target, vae_encoder, vae_decoder, G_m_layer, G_g_layer, final_conv1, final_conv2, final_prediction, atrous_layers, bands, loss_layer, model):
    xc_src, xg_src, xp_src, y_true = x_source
    xc_tgt, xg_tgt, xp_tgt = x_target

    with tf.GradientTape() as tape:
        # Encode
        z_mean_src, z_log_var_src, z_src = vae_encoder(xc_src)
        z_mean_tgt, z_log_var_tgt, z_tgt = vae_encoder(xc_tgt)
        
        # Gating
        gate_weights_src = G_m_layer(z_src)
        gate_weights_tgt = G_m_layer(z_tgt)
        z_modulated_src = tf.multiply(z_src, gate_weights_src)
        z_modulated_tgt = tf.multiply(z_tgt, gate_weights_tgt)

        # Atrous convolution
        atrous_features_src = [tf.reduce_mean(layer(xg_src), axis=[1, 2]) for layer in atrous_layers]
        atrous_features_tgt = [tf.reduce_mean(layer(xg_tgt), axis=[1, 2]) for layer in atrous_layers]

        xg_combined_src = tf.concat(atrous_features_src, axis=-1)
        xg_combined_tgt = tf.concat(atrous_features_tgt, axis=-1)

        gate_global_src = G_g_layer(xg_combined_src)
        gate_global_tgt = G_g_layer(xg_combined_tgt)

        xg_modulated_src = tf.multiply(xg_combined_src, gate_global_src)
        xg_modulated_tgt = tf.multiply(xg_combined_tgt, gate_global_tgt)

        # Coral features
        xp_src_flat = tf.reshape(xp_src, (tf.shape(xp_src)[0], -1))
        xp_tgt_flat = tf.reshape(xp_tgt, (tf.shape(xp_tgt)[0], -1))
        coral_src = tf.concat([xp_src_flat, z_modulated_src, xg_modulated_src], axis=-1)
        coral_tgt = tf.concat([xp_tgt_flat, z_modulated_tgt, xg_modulated_tgt], axis=-1)

        # Prediction head
        z_reshaped_src = tf.reshape(z_modulated_src, (-1, 1, 1, tf.shape(z_modulated_src)[-1]))
        xg_reshaped_src = tf.reshape(xg_modulated_src, (-1, 1, 1, tf.shape(xg_modulated_src)[-1]))
        xp_reshaped_src = tf.reshape(xp_src, (-1, 1, 1, bands))

        concat_input_src = tf.concat([xp_reshaped_src, z_reshaped_src, xg_reshaped_src], axis=-1)

        x = final_conv1(concat_input_src)
        x = final_conv2(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        p_pred = final_prediction(x)

        # VAE Reconstruction
        vae_output = vae_decoder(z_src)

        # Loss
        total_loss = compute_total_loss(xc_src, vae_output, z_mean_src, z_log_var_src, y_true, p_pred, coral_src, coral_tgt)
        
    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # print("Checking gradients:")
    # if any([g is None for g in gradients]):
        # print("⚠️ Some gradients are None!")

 

    return total_loss


def main():
    tf.config.experimental.list_physical_devices('GPU')
    seed = 42
    set_random_seeds(seed)

    weight_b6, weight_b7, weight_b8 = 0.7, 0.13, 0.17
    patch_size = 11
    patch_size_global = 35
    bands = 9
    bands_context = bands
    epoch = 100
    patience_max=5
    batch_size = 1024
    latent_dim = 30
    training=1
    choice = 'tunisia9'
    country = "Tunisia"
    input_raster = f'{choice}.tif'
    output_raster = f'{choice}_processed.tif'
    plots_path = f'plots_{choice}'
    ins_population_csv = 'ins_population.csv'
    target_raster_path = f"POP_{country}.tif"
    model_option = "GAGCoral"
    output_prediction=f'{choice}_popVAE_full_Huber_{model_option}_{batch_size}_{latent_dim}_predicted.tif'
    weights=f"best_weights_popVA_full_Huber_{choice}_{model_option}_{batch_size}_{latent_dim}.h5"
    json=f"model_architecture_popVAE_Huber_{model_option}_{batch_size}_{latent_dim}.json"
    checkpoint_file=f"checkpoint_popVAE_Huber__{choice}_{country}_{model_option}_{batch_size}_{latent_dim}.txt"
    
    input_data, profile = data.preprocess_raster_compososite(input_raster, output_raster,weight_b6,weight_b7,weight_b8, bands+2)
    if country== "Tunisia":
        district_masks = data.load_district_masks(country,24)
    else:
        district_masks = data.load_district_masks(country,8)

    for i, mask in enumerate(district_masks):
        height, width = mask.shape
        print(f"Width: {width}, Height: {height}")
        break
    ins_population = data.load_ins_population_data(ins_population_csv)
    target_data, profile = data.load_and_preprocess_target_data(target_raster_path)
    gc.collect()

    target_input_data, _ = data.load_and_preprocess_target_data("france9.tif")
    input_data_target,_ = data.preprocess_raster_compososite("france9.tif", "france9_processed.tif", weight_b6, weight_b7, weight_b8, bands + 2)
    bands=bands-1
    bands_context=bands

    model = popVAE.create_vae_pipeline_model_simple(patch_size, patch_size_global, latent_dim, bands, bands_context)

    trainable_params = get_trainable_params(model)
    ratio_dataset = (height * width) / (trainable_params * 10)
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Dataset-to-parameter ratio: {ratio_dataset}")

    model_json = model.to_json()
    with open(json, 'w') as json_file:
        json_file.write(model_json)

    all_indices, height, width = popVAE.get_all_indices(ratio_dataset, input_data, patch_size_global, bands)
    input_data_reshaped = input_data.reshape((height, width, bands))

    train_indices, temp_indices = train_test_split(all_indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    training_generator = popVAE.DataGenerator(train_indices, input_data_reshaped, target_data, patch_size, patch_size_global, bands, bands_context, batch_size)
    validation_generator = popVAE.DataGenerator(val_indices, input_data_reshaped, target_data, patch_size, patch_size_global, bands, bands_context, batch_size)
    test_generator = popVAE.DataGenerator(test_indices, input_data_reshaped, target_data, patch_size, patch_size_global, bands, bands_context, batch_size)


    target_input_data_reshaped = input_data_target.reshape((input_data_target.shape[0], input_data_target.shape[1], bands))
    target_indices, _, _ = popVAE.get_all_indices(1.0, target_input_data_reshaped, patch_size_global, bands)
    target_generator = popVAE.TargetDataGenerator(target_indices, target_input_data_reshaped, patch_size, patch_size_global, bands, bands_context, batch_size)

    

    try:
        model.load_weights(weights)
    except:
        print("No best weights found. Starting training from scratch.")
    if (training==1):
        steps_validation = len(validation_generator)
        best_val_loss = float('inf')

        # Create the loss layer ONCE outside of the loop
        num_features = bands + latent_dim + 8 * 4
        loss_layer = popVAE.VAELossLayer(patch_size, bands_context, num_features)

        # Predefine some layers outside to avoid re-accessing inside training step
        vae_encoder = model.get_layer("vae_encoder")
        vae_decoder = model.get_layer("vae_decoder")
        G_m_layer = model.get_layer("G_m")
        G_g_layer = model.get_layer("G_g")
        final_conv1 = model.get_layer('population_prediction_conv1')
        final_conv2 = model.get_layer('population_prediction_conv2')
        final_prediction = model.get_layer('population_prediction')

        atrous_layers = [model.get_layer(f'atrous_conv_r{r}') for r in [1, 3, 11, 17]]

        # ➤ Epoch training
        steps_per_epoch = len(training_generator)
        steps_per_epoch_target = len(target_generator)
        patience=0
        for epoch_idx in range(epoch):

            print(f"\nEpoch {epoch_idx + 1}/{epoch}")
            
            # for step in range(steps_per_epoch):
            step_target_domain=0
            for step in range(steps_per_epoch):
                x_source, _ = training_generator[step]
                
                # Commented out: we no longer get x_target from target_cycle
                if (step % steps_per_epoch_target == 0):
                    step_target_domain=step_target_domain+1
                else:
                    step_target_domain=0

                x_target, _ = target_generator[step_target_domain]
                
                # ➤ Profile memory usage per step
                # mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)
                loss = train_step(x_source, x_target, vae_encoder, vae_decoder, G_m_layer, G_g_layer, final_conv1, final_conv2, final_prediction, atrous_layers, bands, loss_layer, model)
                # loss = tf.constant(0.0)

                del x_source
                del x_target

                # mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)

                # Create the line to print and write
                line = (
                    f"Epoch {epoch_idx + 1}/{epoch} | Step {step + 1}/{steps_per_epoch} "
                    f"- Loss: {loss.numpy():.4f}")
                print("\r" + line, end='')  # still print it live
                if (step + 1) == 1 or (step + 1) % 10 == 0:
                    with open('training_log.txt', 'a') as f:
                        f.write(line + '\n')        # also write it to file

            print("")  # End of epoch

            # ➤ Validation phase
            print("\nValidation start...")
            
            organizing = 'organizing.txt'
            while True:
                
                with open(organizing, 'r') as f:
                    content = f.read().strip()

                if content == '':
                    with open(organizing, 'w') as f:
                        f.write('wait')
                    print("organizing file was empty. Wrote 'wait' and exiting.")
                    break
                elif content != '':
                    time.sleep(5)
                    
            coral_tgt_batches = []
            step_target_domain=-1
            for step in range(steps_validation):
                step_target_domain += 1
                if (step % steps_per_epoch_target == 0):    
                    step_target_domain = 0


                line = (f"Step {step + 1}/{steps_validation}")
                print("\r" + line, end='')  # still print it live

                inputs, _ = target_generator[step_target_domain]
                
                X_c_batch, X_g_batch, X_p_batch = inputs

                # Predict encoding
                z_batch = vae_encoder.predict(X_c_batch, verbose=0)[2]
                z_flattened = tf.reshape(z_batch, (tf.shape(z_batch)[0], -1))
                gate_weights = G_m_layer(z_flattened)
                z_modulated = tf.multiply(z_flattened, gate_weights)
                z_batch_reshaped = tf.reshape(z_modulated, (-1, 1, 1, latent_dim))

                atrous_features = [
                    tf.reduce_mean(layer(X_g_batch), axis=[1, 2]) for layer in atrous_layers
                ]
                xg_combined = tf.concat(atrous_features, axis=-1)
                gate_global = G_g_layer(xg_combined)
                xg_modulated = tf.multiply(xg_combined, gate_global)
                xg_batch = tf.reshape(xg_modulated, (-1, 1, 1, xg_modulated.shape[-1]))

                X_p_reshaped = tf.reshape(X_p_batch, (-1, 1, 1, X_p_batch.shape[-1]))

                coral_tgt_batch = tf.concat([X_p_reshaped, z_batch_reshaped, xg_batch], axis=-1)
                coral_tgt_batch = tf.reshape(coral_tgt_batch, (tf.shape(coral_tgt_batch)[0], -1))
                coral_tgt_batches.append(coral_tgt_batch)

            coral_tgt_stacked = tf.concat(coral_tgt_batches, axis=0)
            del coral_tgt_batches
            coral_tgt_stacked_np = coral_tgt_stacked.numpy()
            del coral_tgt_stacked
            coral_tgt_stacked_np = coral_tgt_stacked_np.reshape((-1, num_features)).astype(np.float32)
            
            # Update validation generator
            validation_generator = popVAE.DataGenerator(
                val_indices,
                input_data_reshaped,
                target_data,
                patch_size,
                patch_size_global,
                bands,
                bands_context,
                batch_size,
                coral_tgt_stacked=coral_tgt_stacked_np
            )
            del coral_tgt_stacked_np
            val_loss = model.evaluate(validation_generator, verbose=0)
            del validation_generator
            print(f"Validation Loss after epoch {epoch_idx + 1}: {val_loss}")
            with open(organizing, 'w') as f:
                f.write('')        
            if val_loss < best_val_loss:
                patience=0
                best_val_loss = val_loss
                model.save_weights(weights)
                line = (f"Epoch {epoch_idx+1}: New best validation loss: {val_loss:.4f}, saving model.")
            else: 
                patience=patience+1
                line = (f"Epoch {epoch_idx+1}: No improvement: {val_loss:.4f}, patience={patience}/{patience_max}.")
                if patience==patience_max:
                    break;

            print(line)
            
            with open('training_log.txt', 'a') as f:
                f.write(line + '\n')        

            gc.collect()
        model.save(weights)
    predicted_image = popVAE.predict_and_reconstruct(model, input_data, profile, output_prediction, patch_size, patch_size_global, bands, bands_context,1000, latent_dim, checkpoint_file)

    print("Evaluating Start")
    ev.calculate_pixel_r2(target_data, predicted_image)


    ev.calculate_district_r2(predicted_image, district_masks, ins_population)
    print ("Evaluation Gate_Atrous_Gate_CORAL latent_dim=",latent_dim)

if __name__ == "__main__":
    main()
