import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add,Input, GlobalAveragePooling2D, Dense, Multiply, Conv2D, Conv2DTranspose, Flatten, Reshape, Lambda, concatenate, multiply
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import rasterio
import random
import os

from tensorflow.keras.initializers import HeNormal,LecunNormal


import gc

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,indices, input_data, target_data, patch_size, bands, bands_context, batch_size):
        self.input_data = input_data
        self.target_data = target_data
        self.patch_size = patch_size
        self.bands = bands
        self.bands_context = bands_context
        self.batch_size = batch_size
        self.height, self.width = input_data.shape[1], input_data.shape[2]
        self.half_patch_size = patch_size // 2
        self.indices = indices
        self.on_epoch_end()

    def __len__(self):
        # Calculate the number of batches per epoch
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_c_batch, X_p_batch, P_gt_batch = self.__data_generation(batch_indices)
        return [X_c_batch,X_p_batch,P_gt_batch], [X_c_batch,P_gt_batch]

    def on_epoch_end(self):
        # Shuffle indices after each epoch if needed
        np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        X_c_batch = []
        X_p_batch = []
        P_gt_batch = []
        
        for i, j in batch_indices:
            patch = self.input_data[i-self.half_patch_size:i+self.half_patch_size+1,j-self.half_patch_size:j+self.half_patch_size+1,: ]
            sum_of_cells = float(tf.reduce_sum(patch).numpy())
            #print("sum_of_cells=",sum_of_cells)
            if(sum_of_cells!=0):
                X_c = patch[:, :, -self.bands_context:]
                X_p = patch[ self.half_patch_size, self.half_patch_size,:]
                P_gt = self.target_data[i, j]
                    
                X_c_batch.append(X_c)
                X_p_batch.append(X_p)
                P_gt_batch.append(P_gt)
        
        X_c_batch = np.array(X_c_batch)
        X_p_batch = np.array(X_p_batch)
        P_gt_batch = np.array(P_gt_batch)
        return X_c_batch, X_p_batch, P_gt_batch
    
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, bands_context, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.bands_context = bands_context

    def call(self, inputs):
        y_true, p_pred = inputs
        total_loss = MeanSquaredError()(y_true, p_pred)
        self.add_loss(total_loss)
        return total_loss

def se_block(input_tensor, reduction_ratio=16):
    channels = input_tensor.shape[-1] #Extract Number of Channels:
    se = GlobalAveragePooling2D()(input_tensor) #Squeeze Operation:The result is a tensor with shape (1, 1, channels),
                                                #which summarizes global information about each channel in the input tensor.
    se = Dense(channels // reduction_ratio, activation='relu')(se) #Excitation Operation: Reduces the number of channels by a factor of reduction_ratio
    se = Dense(channels, activation='sigmoid')(se) # Excitation Operation: Restores the number of channels to the original size
    se = Reshape((1, 1, channels))(se) # This allows it to be used for element-wise multiplication with the original input tensor.
    return Multiply()([input_tensor, se]) # This operation scales each channel of the input tensor according to its importance score

def cbam_block(input_feature,kernel_attention):
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attention = Conv2D(filters=1, kernel_size=kernel_attention, strides=1, padding='same', activation='sigmoid')(concat)
    return multiply([input_feature, attention])

def conv_block(x, filters, kernel_size=3, padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding, activation='relu', kernel_initializer=HeNormal())(x)
    return x

def residual_block(x, filters):
    shortcut = x
    
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    
    # Match dimensions if needed (not required here as we use 'same' padding)
    if K.int_shape(shortcut)[-1] != K.int_shape(x)[-1]:
        x = Conv2D(K.int_shape(shortcut)[-1], (1, 1), padding='same', kernel_initializer=HeNormal())(x)
    
    x = Add()([x, shortcut])
    return x

def residual_dense_block(x, units, activation='relu'):
    shortcut = x
    x = Dense(units, activation=activation, kernel_initializer=HeNormal())(x)
    x = Dense(units, activation=activation, kernel_initializer=HeNormal())(x)
    return Add()([x, shortcut])

def create_vae_pipeline_model_simple(patch_size, latent_dim, bands, bands_context):
    # VAE Encoder
    xc = Input(shape=(patch_size, patch_size, bands_context), name='input_xc')
    x = conv_block(xc, 16)  #  filters from 8 to 16
    x = residual_block(x, 32)  # filters from 16 to 32

    x = Flatten(name='encoder_flatten')(x)
    x = Dense(32, activation='relu', kernel_initializer=HeNormal(), name='encoder_dense' )(x)  #  Dense layer size from 16 to 32
    
    # Prediction Pipeline
    input_xp = Input(shape=(1, 1, bands), name='input_xp')

    # concatenate with input_xp
    x_reshaped = Reshape((1, 1, 32))(x)  # Reshape z to match spatial dimensions of input_xp

    concatenated = concatenate([tf.reshape(input_xp, (-1, 1, 1, input_xp.shape[-1])), x_reshaped], axis=-1)

    # Apply fewer convolutional layers
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(concatenated)  # Reduced filters
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # Reduced filters


    # Global Average Pooling to reduce to scalar
    x = GlobalAveragePooling2D()(x)

    # Output scalar for p_pred
    p_pred = Dense(1, activation='softplus', name='population_prediction')(x)

    # Define models
    y_true = Input(shape=(1,), name='true_population')
    vae_loss_layer = VAELossLayer(patch_size, bands_context, name='vae_loss')(
        [y_true, p_pred]
    )

    vae_pipeline_model = Model(inputs=[xc, input_xp, y_true], outputs=[p_pred], name='vae_pipeline_model')
    vae_pipeline_model.add_loss(vae_loss_layer)

    # Define the learning rate
    learning_rate = 0.001

    # Instantiate the optimizer with the custom learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model with the custom optimizer
    vae_pipeline_model.compile(optimizer=optimizer, loss=None)

    return vae_pipeline_model

def predict_and_reconstruct(model, input_data, profile, output_raster, patch_size, bands, bands_context, batch_size_p, checkpoint_file='checkpoint_popCNN.txt'):
    half_patch_size = patch_size // 2
    _, height, width = input_data.shape[2], input_data.shape[0], input_data.shape[1]
    input_data_reshaped = input_data.reshape((height, width, bands))

    # Check if the output raster file exists
    if os.path.exists(output_raster):
        
        print("Load the existing predicted population density")
        with rasterio.open(output_raster) as src:
            predicted_population_density = src.read(1)
    else:
        print("Initialize a zero array for the predicted population density")
        predicted_population_density = np.zeros((height, width))

    # Check if a checkpoint file exists to resume from the last position
    if os.path.exists(checkpoint_file):
        print("Checkpoint file exists to resume from the last position")
        with open(checkpoint_file, 'r') as f:
            last_i, last_j = map(int, f.read().strip().split(','))
        
    else:
        print("Checkpoint file not existing to resume from the last position")
        last_i, last_j = half_patch_size, half_patch_size  # Start from the first valid position


    # Initialize lists to hold the batch data
    batch_X_c = []
    batch_X_p = []
    batch_positions = []

    # Loop over the entire input image
    # Loop over the entire input image, resuming from the last position
    for i in range(last_i, height - half_patch_size):
        for j in range(last_j if i == last_i else half_patch_size, width - half_patch_size):
            print(f"Processing (i,j)=({i},{j})")
            # Extract the patch centered at (i, j)
            patch = input_data_reshaped[i - half_patch_size:i + half_patch_size + 1, j - half_patch_size:j + half_patch_size + 1, :]
            sum_of_cells = float(tf.reduce_sum(patch).numpy())
            #print("sum_of_cells=",sum_of_cells)
            if(sum_of_cells!=0):
                X_c = patch[:, :, -bands_context:]
                X_p = patch[half_patch_size, half_patch_size, :]

                # Append the data to the batch lists
                batch_X_c.append(X_c)
                batch_X_p.append(X_p)
                batch_positions.append((i, j))
            else:
                print("Skipping")
            # If batch is full, predict and reset batch
          
            if (len(batch_X_c) == batch_size_p) or ((j==(width - half_patch_size-1)) and  (i==(height - half_patch_size-1))) :
                print ("Batch collected")
                samples=batch_size_p
                if (len(batch_X_c) != batch_size_p):
                    samples=len(batch_X_c) 
                batch_X_c_np = np.array(batch_X_c).reshape(samples, patch_size, patch_size, bands_context)
                batch_X_p_np = np.array(batch_X_p).reshape(samples, 1, 1, bands)
                # Predict the population density for the batch
                batch_predicted_pop = model.predict([batch_X_c_np, batch_X_p_np, np.zeros((samples,))])

                # Assign predictions to the correct positions in the output array
                for (i_pos, j_pos), pred in zip(batch_positions, batch_predicted_pop):
                    print(f"Position ({i_pos},{j_pos})={pred}")
                    predicted_population_density[i_pos, j_pos] = pred

                # Clear the batch lists
                batch_X_c.clear()
                batch_X_p.clear()
                batch_positions.clear()


                # Save the predicted population density to the output raster file
                profile.update(count=1)  # Update profile to single band
                with rasterio.open(output_raster, 'w', **profile) as dst:
                    dst.write(predicted_population_density.astype(rasterio.float32), 1)

            # Save the current position to the checkpoint file
            
            with open(checkpoint_file, 'w') as f:
                f.write(f"{i},{j}")  
            print("Saving")
                    
                    
            tf.keras.backend.clear_session()

    predicted_population_density = np.expm1(predicted_population_density)  # Use expm1 to revert log1p
    
    # Save the predicted image
    profile.update(count=1)  # Update profile to single band
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(predicted_population_density.astype(rasterio.float32), 1)
    
    print(f"Predicted raster saved to {output_raster}.")
    
    return predicted_population_density

def get_all_indices(ratio_dataset, input_data, patch_size, bands):
    half_patch_size = patch_size // 2
    _, height, width = input_data.shape[2], input_data.shape[0], input_data.shape[1]
    
    if height > patch_size and width > patch_size:
        all_indices = [(i, j) for i in range(half_patch_size, height - half_patch_size)
                               for j in range(half_patch_size, width - half_patch_size)]
    else:
        raise ValueError("Patch size is too large for the input data dimensions.")
    
    # Calculate the number of random samples (1/ratio_dataset of the total)
    subset_size = int(len(all_indices) / ratio_dataset)  # Convert the result to an integer
    print ("subset_size:", subset_size)
    # Get a random sample of 1/ratio_dataset of the all_indices
    random_indices = random.sample(all_indices, subset_size)
    
    return random_indices, height, width
