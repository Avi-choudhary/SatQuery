import ee

# Initialize Earth Engine with your project ID
ee.Initialize(project="cogent-tract-507708-i6")

# Define your Area of Interest (matching your Sentinel-2 geometry)
geom = ee.Geometry.Polygon([[
    [77.05, 28.55],
    [77.35, 28.55],
    [77.35, 28.85],
    [77.05, 28.85],
    [77.05, 28.55],
]])

# Load Sentinel-1 GRD collection
s1_collection = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(geom)
    .filterDate("2026-01-01", "2026-02-01")
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .select(["VV", "VH"])
)

# Create a composite image (mean) and clip to geometry
image = s1_collection.mean().clip(geom)

# Configure the export task to Google Drive
task = ee.batch.Export.image.toDrive(
    image=image,
    description="delhi_20260112_S1",
    folder="GEE_Exports",
    region=geom,
    scale=10,
    crs="EPSG:32643",
    maxPixels=1e9,
)

task.start()
print("Sentinel-1 export task started successfully!")