import rasterio
import pandas as pd
import numpy as np

def load_district_masks(country,num_districts):
    district_masks = []
    for district in range(num_districts):
        district_mask_path = f'{country}_Regions/{country}_Region_{district}.tif'
        with rasterio.open(district_mask_path) as src:
            district_mask = src.read(1)
        district_masks.append(district_mask)
    return district_masks

def load_ins_population_data(csv_path):
    ins_data = pd.read_csv(csv_path)
    ins_population = ins_data.set_index('code_gov')['population'].to_dict()
    return ins_population

def preprocess_raster_compososite(input_raster, output_raster, b6, b7, b8, n):
    with rasterio.open(input_raster) as src:
        data = src.read()
        profile = src.profile

        no_data_value = -3.4028235e+38
        data[np.isclose(data, no_data_value, atol=1e-5)] = np.nan
        data = np.nan_to_num(data, nan=0.0)
        data[np.isinf(data)] = 0.0  # just in case
        weighted_band_6 = data[n-3] * b6
        weighted_band_7 = data[n-2] * b7
        weighted_band_8 = data[n-1] * b8

        composite_band = weighted_band_6 + weighted_band_7 + weighted_band_8

        data[n-3] = composite_band
        data = np.delete(data, [n-2, n-1], axis=0)

        profile.update(count=n-2, nodata=0.0)  # ✅ Make sure no nodata is preserved

        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32))

    # Reload processed data for model input
    with rasterio.open(output_raster) as src:
        input_data = src.read()
        input_data = np.moveaxis(input_data, 0, -1)

    print(f"Data preprocessing complete for {input_raster}. Preprocessed raster saved to {output_raster}.")
    print(f"Sanity check: min={input_data.min()}, max={input_data.max()}, has_nan={np.isnan(input_data).any()}")
   
    
    return input_data, profile

  
def preprocess_raster_non_compososite(input_raster, output_raster, weight_preparatory=0, weight_institute=0):

    
    adjustment_factor = 1 / (1 - (weight_preparatory + weight_institute))

    with rasterio.open(input_raster) as src:
        data = src.read()
        profile = src.profile
        no_data_value = -3.4028230607370965e+38
        data[data == no_data_value] = np.nan
        data = np.nan_to_num(data, nan=0.0)
        
        print(data.shape)

        # Apply the adjustment to the last band (assuming it's the student population band)
        data[-1, :, :] *= adjustment_factor
        
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32))
    
        with rasterio.open(output_raster) as src:
            input_data = src.read()
            input_data = np.moveaxis(input_data, 0, -1)
        
    print(f"Data preprocessing complete for {input_raster}. Preprocessed raster saved to {output_raster}.")
    return input_data

def load_and_preprocess_target_data(target_raster_path):
    with rasterio.open(target_raster_path) as src:
        target_data = src.read(1)
        no_data_value = -999.0
        target_data[target_data == no_data_value] = np.nan
        target_data = np.log1p(target_data)
        target_data = np.nan_to_num(target_data, nan=0.0)

        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=0.0)
        
        preprocessed_target_raster = 'POP_preprocessed.tif'
        with rasterio.open(preprocessed_target_raster, 'w', **profile) as dst:
            dst.write(target_data.astype(rasterio.float32), 1)
    
    print(f"Target data preprocessing complete. Preprocessed raster saved to {preprocessed_target_raster}.")
    print(target_data.shape)
    return target_data, profile

def load_and_preprocess_target_data_without_log(target_raster_path):
    with rasterio.open(target_raster_path) as src:
        target_data = src.read(1).astype('float32')
        target_data[target_data < 0] = np.nan
        target_data = np.nan_to_num(target_data, nan=0.0)

        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=0.0)

    print(target_data.shape)
    return target_data, profile
