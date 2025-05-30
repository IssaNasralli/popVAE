import arcpy
from arcpy.sa import *
import os

# Set environment settings
arcpy.env.workspace = r"F:\Projects_tunisia\ArcGIS\Building"
arcpy.env.overwriteOutput = True

# Check out the ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension("Spatial")

# Input raster
input_raster = r"F:\Projects_tunisia\data\raw\GeoTif\Tunisia_WSF2019_WGS_84_32N.tif"

# Temporary workspace for intermediate files
temp_workspace = r"F:\Projects_tunisia\ArcGIS\Building\temp"

# Create temporary workspace if it doesn't exist
if not arcpy.Exists(temp_workspace):
    arcpy.CreateFolder_management(arcpy.env.workspace, "temp")

# Output raster
output_raster = r"F:\Projects_tunisia\ArcGIS\Building\Tunisia_WSF2019_WGS_84_32N_0To1.tif"

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
    print(f"Normalization complete for {out_raster}.")

# Split the raster into smaller parts
num_tiles = 4  # Number of tiles to split into
raster = arcpy.Raster(input_raster)
tile_width = raster.extent.width / num_tiles
tile_height = raster.extent.height / num_tiles

# List to store paths of normalized tiles
normalized_tiles = []

for i in range(num_tiles):
    for j in range(num_tiles):
        xmin = raster.extent.XMin + i * tile_width
        ymin = raster.extent.YMin + j * tile_height
        xmax = xmin + tile_width
        ymax = ymin + tile_height

        tile_extent = arcpy.Extent(xmin, ymin, xmax, ymax)
        tile_raster = os.path.join(temp_workspace, f"tile_{i}_{j}.tif")

        # Clip the raster to the tile extent
        arcpy.management.Clip(in_raster=input_raster, rectangle=f"{xmin} {ymin} {xmax} {ymax}", out_raster=tile_raster)

        # Normalize the tile
        normalized_tile = os.path.join(temp_workspace, f"normalized_tile_{i}_{j}.tif")
        normalize_raster(tile_raster, normalized_tile)

        normalized_tiles.append(normalized_tile)

# Mosaic the normalized tiles back together
arcpy.management.MosaicToNewRaster(
    input_rasters=normalized_tiles,
    output_location=os.path.dirname(output_raster),
    raster_dataset_name_with_extension=os.path.basename(output_raster),
    pixel_type="32_BIT_FLOAT",
    number_of_bands=1,
    mosaic_method="BLEND"
)

print("Mosaicking complete.")

# Cleanup temporary files
arcpy.Delete_management(temp_workspace)
print("Temporary files cleaned up.")
