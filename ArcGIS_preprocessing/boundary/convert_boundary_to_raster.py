import arcpy
import os

# Set the absolute path to the directory where your script and shapefile reside
script_dir = "F:\\A Projects_tunisia\\ArcGIS\\MAsk"

# Set the path to the POP.tif file
pop_tif_path = os.path.join(script_dir, "POP.tif")

# Get the extent and cell size of the POP.tif file
desc = arcpy.Describe(pop_tif_path)
pop_extent = desc.extent
pop_cellsize = desc.meanCellWidth

# Set environment settings
arcpy.env.workspace = script_dir  # Set workspace to the directory of the script
arcpy.env.overwriteOutput = True
arcpy.env.extent = pop_extent  # Set the extent to match the POP.tif file
arcpy.env.cellSize = pop_cellsize  # Set the cell size to match the POP.tif file

# Set the input shapefile (district boundaries)
in_features = os.path.join(script_dir, "Tunisia_Regions_Project.shp")

# Ensure the input shapefile exists
if not arcpy.Exists(in_features):
    raise FileNotFoundError(f"The shapefile {in_features} does not exist.")

# Set the output directory for the created TIFF files
output_dir = os.path.join(script_dir, "Tunisia_Regions")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over each district
for district_id in range(24):
    try:
        # Set the SQL expression to select the current district
        sql_expression = f"FID = {district_id}"
        
        # Create a feature layer for the current district
        temp_layer = f"temp_layer_{district_id}"
        arcpy.management.MakeFeatureLayer(in_features, temp_layer, sql_expression)
        
        # Check if the layer was created
        if not arcpy.Exists(temp_layer):
            raise ValueError(f"Failed to create feature layer for District {district_id}.")
        
        # Set the output raster dataset for the current district
        out_raster = os.path.join(output_dir, f"Tunisia_Region_{district_id}.tif")
        
        # Delete the output raster if it already exists
        if arcpy.Exists(out_raster):
            arcpy.management.Delete(out_raster)
        
        # Convert the polygon feature layer to a raster dataset with pixel values set to 1
        temp_raster = os.path.join(output_dir, f"temp_{district_id}.tif")
        arcpy.conversion.PolygonToRaster(temp_layer, "FID", temp_raster, 
                                         "CELL_CENTER", "", pop_cellsize)
        
        # Update the raster to set all non-NoData values to 1
        con_result = arcpy.sa.Con(arcpy.sa.Raster(temp_raster) >= 0, 1)
        con_result.save(out_raster)
        
        # Clean up the temporary raster
        arcpy.management.Delete(temp_raster)
        
        print(f"Mask TIF created for District {district_id}.")
    
    except Exception as e:
        print(f"Error processing District {district_id}: {e}")
    
    finally:
        # Clean up temporary feature layer
        if arcpy.Exists(temp_layer):
            arcpy.management.Delete(temp_layer)

print("All districts processed.")
