import arcpy
arcpy.env.workspace = r"F:\Projects_tunisia\ArcGIS\Building"

# Define the source and target coordinate systems
target_cs = arcpy.SpatialReference(32632)  # WGS 1984 UTM Zone 32N
source_cs = arcpy.SpatialReference(4326)   # WGS 1984 

# Path to the WorldPop dataset
worldpop_path = r"F:\Projects_tunisia\data\raw\GeoTif\Tunisia_WSF2019.tif"

# Reproject the WorldPop dataset
reprojected_worldpop = worldpop_path.replace(".tif", "_WGS_84_32N.tif")
arcpy.ProjectRaster_management(
    in_raster=worldpop_path,
    out_raster=reprojected_worldpop,
    out_coor_system=target_cs
)
