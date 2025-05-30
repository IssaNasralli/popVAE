Step 1: Add the Road Network Shapefile to ArcGIS Pro

    Open ArcGIS Pro and start a new project or open an existing one.
    In the Catalog pane, navigate to your road network shapefile and add it to the map by dragging it onto the map view.

Step 2: Open the Attribute Table

    Right-click on the road network layer in the Contents pane.
    Select Attribute Table to open the attribute table of the shapefile.

Step 3: Use the Select by Attributes Tool

    Go to the Analysis tab on the ribbon.
    Click on Select and then Select by Attributes.

Step 4: Construct the Query to Select Specific Road Types

    In the Select by Attributes window, select the road network layer.

    In the SQL expression box, construct a query to select the desired road types. For example:

    sql

    "highway" IN ('motorway', 'primary', 'secondary', 'tertiary')

    This query assumes that the column containing the road type information is named road_type. Adjust the column name according to your attribute table.

    Click Apply to execute the query. This will select the rows that match the specified road types.

Step 5: Create a New Layer from the Selected Features

    With the roads of interest selected, right-click on the road network layer in the Contents pane.
    Choose Data > Export Features.
    In the Export Features dialog box:
        Set the output location and name for the new shapefile.
        Ensure that the option Selected features is checked.
    Click OK to create the new shapefile containing only the selected road types.