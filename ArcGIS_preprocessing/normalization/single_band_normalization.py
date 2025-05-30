import arcpy
from arcpy.sa import *

# Set environment settings
arcpy.env.workspace = "M:\\AA France\\Final_processing"
arcpy.env.overwriteOutput = True

# Check out the ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension("Spatial")

# Input raster
input_raster = "Secondary_resamled"
# Output raster
output_raster = "Secondary_resamled_normalized.tif"
# Function to normalize raster
def normalize_raster(in_raster, out_raster):
    # Convert to 32-bit float
    float32_raster = Float(in_raster)

    # Set NoData values to 0
    float32_raster = SetNull(float32_raster == float32_raster.noDataValue, float32_raster)
    print("Set NoData values to 0.")

    # Normalize raster
    normalized_raster = (float32_raster - float32_raster.minimum) / (float32_raster.maximum - float32_raster.minimum)
    normalized_raster.save(out_raster)
    print("Normalization complete.")

# Run normalization
normalize_raster(input_raster, output_raster)
