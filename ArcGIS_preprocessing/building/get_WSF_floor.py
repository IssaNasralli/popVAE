import arcpy

# Define the input raster
raster_tif = r"F:\Projects_tunisia\data\raw\GeoTif\WSF3D_V02_BuildingVolume.tif"

# Set the workspace environment
arcpy.env.workspace = r"F:\Projects_tunisia\ArcGIS\Building"

# Define the boundary shapefile to clip the image
boundary_shapefile = r"F:\Projects_tunisia\data\raw\shapefile\boundaries\Tunisia\Tunisia_Boundary.shp"

# Define the output clipped raster
output_clipped_raster = r"F:\Projects_tunisia\ArcGIS\Building\Tunisia_floor.tif"

# Perform ExtractByMask
clipped_raster = arcpy.sa.ExtractByMask(raster_tif, boundary_shapefile)

# Process the clipped raster
clipped_raster = clipped_raster / 4000 / 3

# Save the output
clipped_raster.save(output_clipped_raster)

print("Clipping and processing completed successfully.")
