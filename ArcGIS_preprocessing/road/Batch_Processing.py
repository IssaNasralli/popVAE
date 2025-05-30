use the Batch Processing tool in ArcGIS Pro to perform snapping operations on your road network. Here’s a guide to help you through the process:
Step 1: Prepare Your Data

Ensure that your filtered road network shapefile is added to your ArcGIS Pro project.
Step 2: Enable Snapping in ArcGIS Pro

    Go to the Edit tab on the ribbon.
    Click on the Snapping dropdown and enable Snapping if it's not already enabled.

Step 3: Create a Snapping Environment

    In the Edit tab, under Snapping, choose Snapping Settings to configure your snapping environment. Make sure the snapping types you need (e.g., Vertex, Edge, End) are enabled.

Step 4: Use the Integrate Tool for Snapping

    Go to the Analysis tab on the ribbon.
    Click on Tools to open the Geoprocessing pane.
    In the Geoprocessing pane, search for the Integrate tool.
    Set the parameters:
        Input Features: Select your road network layer.
        Cluster Tolerance: Set a distance tolerance within which features will snap together. This value depends on the precision required for your project.