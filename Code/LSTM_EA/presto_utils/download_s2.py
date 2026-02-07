import ee
import pandas as pd
import os

# --- 1. Configuration ---
# Output file path
OUTPUT_FILE = "../Data/s2_pixels/s2_presto_inputs.csv"

# Date Range (Adjust to match your study period)
START_DATE = '2017-01-01'
END_DATE = '2023-12-31'

# Station Coordinates (AmeriFlux Mead, NE sites)
# PLEASE VERIFY these match your specific tower metadata exactly!
STATIONS = {
    'US-Ne1': [-96.4766, 41.1651],  # Irrigated Continuous Maize
    'US-Ne2': [-96.4701, 41.1649],  # Irrigated Maize/Soy Rotation
    'US-Ne3': [-96.4397, 41.1797]   # Rainfed Maize/Soy Rotation
}

# Bands required by Presto
# B2, B3, B4 (Visible), B8 (NIR), B11, B12 (SWIR) are the most critical.
BANDS = [
    'B1',   # Aerosols
    'B2',   # Blue
    'B3',   # Green
    'B4',   # Red
    'B5',   # Red Edge 1
    'B6',   # Red Edge 2
    'B7',   # Red Edge 3
    'B8',   # NIR
    'B8A',  # Red Edge 4 / Narrow NIR
    'B9',   # Water Vapor
    'B11',  # SWIR 1
    'B12',  # SWIR 2
    'QA60'  # Cloud Mask
]
# --- 2. Helper Functions ---

def mask_s2_clouds(image):
    """
    Standard Sentinel-2 Cloud Masking using QA60 band.
    Bits 10 and 11 are clouds and cirrus, respectively.
    """
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
             qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    
    # Return the masked image, scaled to 0-10000 (standard S2 reflectance)
    # Presto expects raw integers (0-10000) or floats (0-1). 
    # We'll keep it as Refl units and normalize later.
    return image.updateMask(mask).select(BANDS).copyProperties(image, ["system:time_start"])

def get_station_data(site_name, coords):
    print(f"Extracting data for {site_name}...")
    point = ee.Geometry.Point(coords)
    
    # Load Harmonized Sentinel-2 (L2A Surface Reflectance)
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(START_DATE, END_DATE) \
        .filterBounds(point) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)) \
        .map(mask_s2_clouds)
    
    # Extract time series at the point (Scale: 10m)
    # getRegion returns: [header, row1, row2, ...]
    data = s2.getRegion(point, 10).getInfo()
    
    # Convert to DataFrame
    header = data[0]
    rows = data[1:]
    
    if not rows:
        print(f"Warning: No data found for {site_name}")
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=header)
    df['Site'] = site_name
    df['latitude'] = coords[1]
    df['longitude'] = coords[0]
    
    # Convert time to readable Date
    df['Date'] = pd.to_datetime(df['time'], unit='ms')
    
    # Drop rows where critical bands are masked (NaN)
    # 'getRegion' might return null for masked pixels
    df = df.dropna()
    
    return df

# --- 3. Main Execution ---

def main():
    # Trigger GEE Authentication
    try:
        ee.Initialize(project= "ee-rouhinmitraucla")
    except Exception:
        print("Authenticating with Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project= "ee-rouhinmitraucla")

    all_data = []
    
    for name, coords in STATIONS.items():
        site_df = get_station_data(name, coords)
        all_data.append(site_df)
    
    # Combine and Save
    if all_data:
        final_df = pd.concat(all_data)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Downloaded {len(final_df)} rows.")
        print(f"Saved to: {OUTPUT_FILE}")
        print("Columns:", list(final_df.columns))
    else:
        print("Failed to retrieve data.")

if __name__ == "__main__":
    main()