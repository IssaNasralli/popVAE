import arcpy
arcpy.env.workspace = r"F:\Projects_tunisia\ArcGIS\Building"
# Define the paths to the input raster files
Tunisia_floor = "Tunisia_floor_WGS_84_32N.tif"
Tunisia_WSF2019_0to1 = "Tunisia_WSF2019_WGS_84_32N_0To1.tif"

# Define the output weighted raster file path
Tunisia_floor_WSF2019_0to1 = "Tunisia_floor_WSF2019_WGS_84_32N_0to1.tif"

try:
    # Check out Spatial Analyst extension license
    arcpy.CheckOutExtension("Spatial")

    # Read the raster datasets
    Tunisia_floor_raster = arcpy.Raster(Tunisia_floor)
    Tunisia_WSF2019_0to1_raster = arcpy.Raster(Tunisia_WSF2019_0to1)

    # Multiply the estimated number of floors with the WSF dataset
    Tunisia_floor_WSF2019_0to1_raster = arcpy.sa.Times(Tunisia_floor_raster, Tunisia_WSF2019_0to1_raster)

    # Save the resulting weighted raster to a new file
    Tunisia_floor_WSF2019_0to1_raster.save(Tunisia_floor_WSF2019_0to1)

    print("Weighted raster file created successfully.")

except arcpy.ExecuteError:
    print(arcpy.GetMessages(2))

except Exception as e:
    print(e)
    