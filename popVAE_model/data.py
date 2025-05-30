import rasterio
import pandas as pd
import numpy as np

def load_district_masks(num_districts=24):
    district_masks = []
    for district in range(num_districts):
        district_mask_path = f'Tunisia_Regions/Tunisia_Region_{district}.tif'
        with rasterio.open(district_mask_path) as src:
            district_mask = src.read(1)
        district_masks.append(district_mask)
    return district_masks

def load_ins_population_data(csv_path):
    ins_data = pd.read_csv(csv_path)
    ins_population = ins_data.set_index('code_gov')['population'].to_dict()
    return ins_population
def preprocess_raster_compososite(input_raster, output_raster,b6,b7,b8):
    with rasterio.open(input_raster) as src:
        data = src.read()
        profile = src.profile
        no_data_value = -3.4028230607370965e+38
        data[data == no_data_value] = np.nan
        data = np.nan_to_num(data, nan=0.0)

        weighted_band_6 = data[9] * b6
        weighted_band_7 = data[10] * b7
        weighted_band_8 = data[11] * b8
        
        composite_band = weighted_band_6 + weighted_band_7 + weighted_band_8
        
        data[9] = composite_band
        
        data = np.delete(data, [10, 11], axis=0)
        
        print("input_data shape,",data.shape)
        profile.update(count=10)

        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32))
    
        with rasterio.open(output_raster) as src:
            input_data = src.read()
            input_data = np.moveaxis(input_data, 0, -1)
        
    print(f"Data preprocessing complete for {input_raster}. Preprocessed raster saved to {output_raster}.")
    return input_data
    
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
