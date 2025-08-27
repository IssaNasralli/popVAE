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
    def __init__(self, indices, input_data, target_data, patch_size, patch_size_global, bands, bands_context, batch_size, coral_tgt_stacked=None):
        self.input_data = input_data
        self.target_data = target_data
        self.patch_size = patch_size
        self.patch_size_global = patch_size_global
        self.bands = bands
        self.bands_context = bands_context
        self.batch_size = batch_size
        self.height, self.width = input_data.shape[1], input_data.shape[2]
        self.half_patch_size = patch_size // 2
        self.half_patch_size_global = patch_size_global // 2
        self.indices = indices
        self.coral_tgt_stacked = coral_tgt_stacked
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_c_batch, X_g_batch, X_p_batch, P_gt_batch = self.__data_generation(batch_indices)

        # Attach the corresponding coral_tgt if available
        if self.coral_tgt_stacked is not None:
            coral_batch = self.coral_tgt_stacked[index * self.batch_size:(index + 1) * self.batch_size]
                        
            return [X_c_batch, X_g_batch, X_p_batch, P_gt_batch, coral_batch], [X_c_batch, P_gt_batch]
        else:
            return [X_c_batch, X_g_batch, X_p_batch, P_gt_batch], [X_c_batch, P_gt_batch]

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        X_c_batch = []
        X_g_batch = []
        X_p_batch = []
        P_gt_batch = []

        for i, j in batch_indices:
            patch = self.input_data[i-self.half_patch_size:i+self.half_patch_size+1,
                                    j-self.half_patch_size:j+self.half_patch_size+1, :]
            patch_global = self.input_data[i-self.half_patch_size_global:i+self.half_patch_size_global+1,
                                           j-self.half_patch_size_global:j+self.half_patch_size_global+1, :]
            sum_of_cells = float(tf.reduce_sum(patch).numpy())
            # if sum_of_cells != 0:
            if 1 > 0:
                X_c = patch[:, :, -self.bands_context:]
                X_g = patch_global[:, :, -self.bands_context:]
                X_p = patch[self.half_patch_size, self.half_patch_size, :]
                P_gt = self.target_data[i, j]

                X_g_batch.append(X_g)
                X_c_batch.append(X_c)
                X_p_batch.append(X_p)
                P_gt_batch.append(P_gt)

        return (
            np.array(X_c_batch),
            np.array(X_g_batch),
            np.array(X_p_batch),
            np.array(P_gt_batch)
        )

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
    
    population_size = len(all_indices)    
    if subset_size > population_size:
        subset_size = population_size  # fallback to full sample    


    print ("subset_size:", subset_size)
    # Get a random sample of 1/ratio_dataset of the all_indices
    random_indices = random.sample(all_indices, subset_size)
    
    return random_indices, height, width


class TargetDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,indices, input_data, patch_size, patch_size_global, bands, bands_context, batch_size):
        self.input_data = input_data
        self.patch_size = patch_size
        self.patch_size_global = patch_size_global
        self.bands = bands
        self.bands_context = bands_context
        self.batch_size = batch_size
        self.height, self.width = input_data.shape[1], input_data.shape[2]
        self.half_patch_size = patch_size // 2
        self.half_patch_size_global = patch_size_global // 2
        self.indices = indices
        self.on_epoch_end()

    def __len__(self):
        # Calculate the number of batches per epoch
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_c_batch,X_g_batch, X_p_batch = self.__data_generation(batch_indices)
        return [X_c_batch,X_g_batch,X_p_batch], [X_c_batch]

    def on_epoch_end(self):
        # Shuffle indices after each epoch if needed
        np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        X_c_batch = []
        X_g_batch = []
        X_p_batch = []
        
        for i, j in batch_indices:
            patch = self.input_data[i-self.half_patch_size:i+self.half_patch_size+1,j-self.half_patch_size:j+self.half_patch_size+1,: ]
            patch_global = self.input_data[i-self.half_patch_size_global:i+self.half_patch_size_global+1,j-self.half_patch_size_global:j+self.half_patch_size_global+1,: ]
            sum_of_cells = float(tf.reduce_sum(patch).numpy())
            #print("sum_of_cells=",sum_of_cells)
            # if(sum_of_cells!=0):
            if(1>0):
                X_c = patch[:, :, -self.bands_context:]
                X_g = patch_global[:, :, -self.bands_context:]
                X_p = patch[ self.half_patch_size, self.half_patch_size,:]
                    
                X_g_batch.append(X_g)
                X_c_batch.append(X_c)
                X_p_batch.append(X_p)
        
        X_g_batch = np.array(X_g_batch)
        X_c_batch = np.array(X_c_batch)
        X_p_batch = np.array(X_p_batch)
        return X_c_batch, X_g_batch, X_p_batch

class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, bands_context, num_features, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.bands_context = bands_context
        self.num_features = num_features

    def call(self, inputs):
        xc, xc_prim, z_mean, z_log_var, y_true, p_pred, coral_src, coral_tgt = inputs
        coral_tgt = tf.ensure_shape(coral_tgt, [None, self.num_features])

        reconstruction_loss = MeanSquaredError()(xc, xc_prim)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        prediction_loss = MeanSquaredError()(y_true, p_pred)
        coral_loss_val = self.coral_loss(coral_src, coral_tgt)
        total_loss = K.mean(reconstruction_loss + kl_loss + prediction_loss + coral_loss_val) 

        self.add_loss(total_loss)
        return total_loss
        
    def coral_loss(self, source, target):
        d = tf.cast(tf.shape(source)[-1], tf.float32)

        # Center the features
        source_c = source - tf.reduce_mean(source, axis=0)
        target_c = target - tf.reduce_mean(target, axis=0)

        # Compute covariance matrices
        cov_source = tf.matmul(tf.transpose(source_c), source_c) / (tf.cast(tf.shape(source)[0], tf.float32) - 1)
        cov_target = tf.matmul(tf.transpose(target_c), target_c) / (tf.cast(tf.shape(target)[0], tf.float32) - 1)

        # Frobenius norm between covariances
        coral = tf.reduce_sum(tf.square(cov_source - cov_target)) / (4 * d * d)
        return coral

def create_vae_pipeline_model_simple(patch_size, patch_size_global, latent_dim, bands, bands_context):
    # VAE Encoder
    xc = Input(shape=(patch_size, patch_size, bands_context), name='input_xc')
    xg = Input(shape=(patch_size_global, patch_size_global, bands_context), name='input_xg')
    x = conv_block(xc, 16)  # Reduced filters from 16 to 8
    x = residual_block(x, 32)  # Reduced filters and simplified residual block

    x = Flatten(name='encoder_flatten')(x)
    x = Dense(32, activation='relu', kernel_initializer=HeNormal(), name='encoder_dense' )(x)  # Reduced Dense layer size
    z_mean = Dense(latent_dim, kernel_initializer=HeNormal(), name='z_mean')(x)
    z_log_var = Dense(latent_dim, kernel_initializer=HeNormal(), name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='latent_sampling')([z_mean, z_log_var])

    # VAE Decoder
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(32, activation='relu', kernel_initializer=HeNormal(), name='decoder_dense1')(decoder_input)  # Reduced Dense layer size
    x = Dense(patch_size * patch_size * 32, activation='relu', kernel_initializer=HeNormal(), name='decoder_dense2')(x)  # Reduced size
    x = Reshape((patch_size, patch_size, 32), name='decoder_reshape')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal(), name='decoder_deconv1')(x)  # Reduced filters
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal(), name='decoder_deconv2')(x)  # Reduced filters
    xc_prim = Conv2DTranspose(bands_context, (3, 3), activation='sigmoid', padding='same', kernel_initializer=LecunNormal(), name='decoder_output')(x)

    # VAE Models
    encoder = Model(xc, [z_mean, z_log_var, z], name='vae_encoder')
    decoder = Model(decoder_input, xc_prim, name='vae_decoder')
    vae_output = decoder(encoder(xc)[2])

    # Prediction Pipeline
    input_xp = Input(shape=(1, 1, bands), name='input_xp')

    # Flatten z and apply gating mechanism G_m
    z_flattened = Flatten()(z)
    gate_weights = Dense(latent_dim, activation='sigmoid', kernel_initializer=HeNormal(), name='G_m')(z_flattened)
    z_modulated = Multiply()([z_flattened, gate_weights])
    z_reshaped = Reshape((1, 1, latent_dim))(z_modulated)


   # Atrous convolution branch on global input
    atrous_rates = [1, 3, 11, 17]
    atrous_features = []

    for rate in atrous_rates:
        feat = Conv2D(8, (3, 3), padding='same', dilation_rate=rate, activation='relu',
                      kernel_initializer=HeNormal(), name=f'atrous_conv_r{rate}')(xg)
        feat = GlobalAveragePooling2D()(feat)  # Reduce spatial dimension
        atrous_features.append(feat)

    xg_combined = concatenate(atrous_features, axis=-1)
    
    # Apply gating mechanism G_g
    gate_global = Dense(xg_combined.shape[-1], activation='sigmoid', kernel_initializer=HeNormal(), name='G_g')(xg_combined)
    xg_modulated = Multiply()([xg_combined, gate_global])  # Element-wise modulation
    xg_reshaped = Reshape((1, 1, -1))(xg_modulated)  # Final shape ready for concat

    concatenated = concatenate([tf.reshape(input_xp, (-1, 1, 1, input_xp.shape[-1])), z_reshaped, xg_reshaped], axis=-1)

    # Apply fewer convolutional layers
    x = Conv2D(16, (3, 3), activation='relu', padding='same',  name='population_prediction_conv1' )(concatenated)  # Reduced filters
    x = Conv2D(8, (3, 3), activation='relu', padding='same',  name='population_prediction_conv2' )(x)  # Reduced filters

    # Global Average Pooling to reduce to scalar
    x = GlobalAveragePooling2D()(x)

    # Output scalar for p_pred
    p_pred = Dense(1, activation='softplus', name='population_prediction')(x)

    # Define models
    y_true = Input(shape=(1,), name='true_population')
    
    
    coral_src = Flatten()(concatenated)
    print("coral_src shape inside model crration:", coral_src.shape)
    # Compute static input shape using known input sizes
    num_features = bands + latent_dim + 8 * 4  # 8 filters × 4 atrous rates
    coral_tgt = Input(shape=(num_features,), name='target_features')   
    vae_loss_layer = VAELossLayer(patch_size, bands_context, num_features, name='vae_loss')(
        [xc, vae_output, z_mean, z_log_var, y_true, p_pred, coral_src, coral_tgt]
    )

    vae_pipeline_model = Model(inputs=[xc, xg, input_xp, y_true, coral_tgt], outputs=[vae_output, p_pred], name='vae_pipeline_model')
    vae_pipeline_model.add_loss(vae_loss_layer)

    # Define the learning rate
    learning_rate = 0.001

    # Instantiate the optimizer with the custom learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model with the custom optimizer
    vae_pipeline_model.compile(optimizer=optimizer, loss=None)

    return vae_pipeline_model

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
  
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def predict_and_reconstruct(model, input_data, profile, output_raster, patch_size, patch_size_global, bands, bands_context, batch_size_p, latent_dim, checkpoint_file):
    half_patch_size_global = patch_size_global // 2
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
        last_i, last_j = half_patch_size_global, half_patch_size_global  # Start from the first valid position

    # Initialize lists to hold the batch data
    batch_X_g = []
    batch_X_c = []
    batch_X_p = []
    batch_positions = []
    max_pred = 0

    # Loop over the entire input image, resuming from the last position
    for i in range(last_i, height - half_patch_size_global):
        for j in range(last_j if i == last_i else half_patch_size_global, width - half_patch_size_global):
            # Extract the patch centered at (i, j)
            patch = input_data_reshaped[i - half_patch_size:i + half_patch_size + 1, j - half_patch_size:j + half_patch_size + 1, :]
            patch_global = input_data_reshaped[i - half_patch_size_global:i + half_patch_size_global + 1, j - half_patch_size_global:j + half_patch_size_global + 1, :]
            sum_of_cells = float(tf.reduce_sum(patch).numpy())
            if sum_of_cells != 0:
                X_c = patch[:, :, -bands_context:]
                X_g = patch_global[:, :, -bands_context:]
                X_p = patch[half_patch_size, half_patch_size, :]

                # Append the data to the batch lists
                batch_X_c.append(X_c)
                batch_X_g.append(X_g)
                batch_X_p.append(X_p)
                batch_positions.append((i, j))

            if (len(batch_X_g) == batch_size_p) or ((j == (width - half_patch_size_global - 1)) and (i == (height - half_patch_size_global - 1))):
                samples = batch_size_p
                if len(batch_X_g) != batch_size_p:
                    samples = len(batch_X_g)
                print(f"Batch collected of {samples} samples at (i,j)=({i},{j})")
                batch_X_c_np = np.array(batch_X_c).reshape(samples, patch_size, patch_size, bands_context)
                batch_X_g_np = np.array(batch_X_g).reshape(samples, patch_size_global, patch_size_global, bands_context)
                batch_X_p_np = np.array(batch_X_p).reshape(samples, 1, 1, bands)
                num_features = bands + latent_dim + 8 * 4 
                # Predict the population density for the batch
                dummy_y_true = np.zeros((samples, 1), dtype=np.float32)
                dummy_coral_tgt = np.zeros((samples, num_features), dtype=np.float32)
                _, batch_predicted_pop = model.predict([batch_X_c_np, batch_X_g_np, batch_X_p_np, dummy_y_true, dummy_coral_tgt ])

                # Assign predictions to the correct positions in the output array
                for (i_pos, j_pos), pred in zip(batch_positions, batch_predicted_pop):
                    predicted_population_density[i_pos, j_pos] = pred
                    if (pred > max_pred):
                        max_pred = pred

                # Clear the batch lists
                batch_X_c.clear()
                batch_X_g.clear()
                batch_X_p.clear()
                batch_positions.clear()

                # Save the predicted population density to the output raster file
                profile.update(count=1)  # Update profile to single band
                with rasterio.open(output_raster, 'w', **profile) as dst:
                    dst.write(predicted_population_density.astype(rasterio.float32), 1)

            # Save the current position to the checkpoint file
            with open(checkpoint_file, 'w') as f:
                f.write(f"{i},{j}")
            tf.keras.backend.clear_session()

    # Define a more reasonable clipping range for 100x100 m areas
    max_value = 6.125  # Upper limit for population density (log-scale)
    min_value = 0  # Lower limit to avoid negative population densities

    # Clip the predicted population density values to avoid overflow and unrealistic predictions
    predicted_population_density = np.clip(predicted_population_density, min_value, max_value)

    # Apply np.expm1 to revert the log transformation safely
    predicted_population_density = np.expm1(predicted_population_density)  # Converts log-population density back to population

    # Save the predicted image
    profile.update(count=1)  # Update profile to single band
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(predicted_population_density.astype(rasterio.float32), 1)

    print(f"Predicted raster saved to {output_raster}.")

    return predicted_population_density
