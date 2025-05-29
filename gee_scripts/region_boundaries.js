// Define a region of interest for Tunisia
var tunisiaRegions = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Tunisia'));

// Specify the Earth Engine Asset path for export
var exportPath = 'tunisiaRegions';
// Add the administrative boundaries to the map
Map.addLayer(tunisiaRegions, {color: 'FF0000'}, 'Tunisia Administrative Regions');

// Zoom to Tunisia
Map.centerObject(tunisiaRegions, 6);

// Export the dataset to Google Drive
Export.table.toDrive({
  collection: tunisiaRegions,
  description: 'Tunisia_Regions',
  folder: exportPath,
  fileFormat: 'SHP', // Change file format as needed (e.g., 'KML', 'GeoJSON')
});
