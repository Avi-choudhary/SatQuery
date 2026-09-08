import ee

# Initialize Earth Engine with your project ID
ee.Initialize(project="cogent-tract-507708-i6")

# Define your Area of Interest (Polygon coordinates)
geom = ee.Geometry.Polygon([[
    [77.05, 28.55],
    [77.35, 28.55],
    [77.35, 28.85],
    [77.05, 28.85],
    [77.05, 28.55],
]])


# Cloud masking function for Sentinel-2
def mask_s2_clouds(image):
  qa = image.select("QA60")
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11
  mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).bitwiseAnd(cirrus_bit_mask).eq(0)
  return image.updateMask(mask).divide(10000.0)


# Load Sentinel-2 Surface Reflectance collection
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(geom)
    .filterDate("2026-01-01", "2026-02-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(mask_s2_clouds)
)

# Create a composite image and select bands (B4, B3, B2, B8)
image = collection.mosaic().select(["B4", "B3", "B2", "B8"])

# Configure the export task to Google Drive
task = ee.batch.Export.image.toDrive(
    image=image,
    description="delhi_20260112_S2",
    folder="GEE_Exports",
    region=geom,
    scale=10,
    crs="EPSG:32643",
    maxPixels=1e9,
)

task.start()
print("Sentinel-2 export task started successfully!")