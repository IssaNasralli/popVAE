// Define a function to stack images for a given region and year
var CollectImagesForRegionAndYear = function(regionName, level, admName, startYear, endYear) {
  // Define a region of interest (ROI) based on the specified parameters
  var country = ee.FeatureCollection('FAO/GAUL/2015/' + level)
    .filter(ee.Filter.eq(admName, regionName));
  var geometry = country.geometry();
  var roi = country.geometry();

  // Iterate over the years
  for (var year = startYear; year <= endYear; year++) {
    // Set the start and end dates based on the year parameter
    var startDate = year + '-01-01';
    var endDate = year + '-12-31';

    // Load Dynamic World data for the specified year
    var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                 .filterDate(startDate, endDate)
                 .filterBounds(geometry);
    var classification = dw.select('label');
    var dwComposite = classification.reduce(ee.Reducer.mode());
    var dwVisParams = {
      min: 0,
      max: 8,
      palette: [
        '#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635', '#DFC35A',
        '#C4281B', '#A59B8F', '#B39FE1'
      ]
    };

    // Display the classified composite on the map
    Map.addLayer(dwComposite.clip(geometry), dwVisParams, 'Classified Composite - ' + regionName + ' ' + year);

if (regionName === 'Sfax') {
  regionName = 'sfax';
} 
    // Export the image to Google Earth Engine assets
    Export.image.toAsset({
      image: dwComposite,
      description: 'DynamicWorld_' + regionName + '_' + year,
      assetId: 'users/aissanasralli/' + regionName + '/' + year + '/original/DynamicWorld_' + regionName + '_' + year,
      scale: 10,
      region: geometry,
      maxPixels: 1e13
    });
  }
};

// Collect images for Tunisia (level0)
CollectImagesForRegionAndYear('Tunisia', 'level0', 'ADM0_NAME', 2016, 2020);

