import ee
import pandas as pd
import os
import math

# --- VERIFICATION ---
print("RUNNING: FINAL DEBUGGED S1 DOWNLOADER")

# --- CONFIGURATION ---
OUTPUT_FILE = "../Data/s1_pixels/s1_presto_inputs.csv"
START_DATE = '2017-01-01'
END_DATE = '2023-12-31'
SITES = {
    'US-Ne1': [-96.4766, 41.1651],
    'US-Ne2': [-96.4701, 41.1649],
    'US-Ne3': [-96.4397, 41.1797]
}

def initialize_gee():
    try:
        ee.Initialize(project='ee-rouhinmitraucla')
        print("Google Earth Engine initialized successfully!")
    except Exception as e:
        print(f"Initialization failed: {e}")
        try:
            ee.Authenticate()
            ee.Initialize(project='ee-rouhinmitraucla')
        except Exception as e2:
            print(f"CRITICAL ERROR: {e2}")
            return False
    return True

def download_s1_data():
    if not initialize_gee():
        return

    print(f"Downloading Daily S1 (Masking Zeros)...")
    all_data = []

    for site_name, coords in SITES.items():
        print(f"Processing {site_name}...")
        point = ee.Geometry.Point(coords)
        
        # Load Collection
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(point) \
            .filterDate(START_DATE, END_DATE) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
        # print(s1)
        def extract_s1(image):
            date_millis = image.date().millis()
            
            # --- FIX: Select ONLY VV/VH first ---
            # This ensures 'image_clean' has exactly 2 bands.
            image_clean = image.select(['VV', 'VH'])
            
            # # Create a 2-band mask (1 where data > 0, 0 where data == 0)
            # mask = image_clean.gt(0)
            
            # # Apply 2-band mask to 2-band image (Perfect Match)
            # image_masked = image_clean.updateMask(mask)
            
            # Reduce region
            stats = image_clean.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=10
            )
            
            return ee.Feature(None, {
                'Site': site_name,
                'Date_Millis': date_millis,
                'VV': stats.get('VV'),
                'VH': stats.get('VH'),
                'Orbit': image.get('orbitProperties_pass')
            })

        # Fetch data
        data = s1.map(extract_s1).getInfo()['features']
        print(f"  Retrieved {len(data)} images")
        
        null_count = 0
        negative_count = 0
        success_count = 0
        error_count = 0
        
        for feat in data:
            props = feat['properties']
            
            # Filter Nulls
            if props['VV'] is not None and props['VH'] is not None:
                try:
                    vv_db = float(props['VV'])
                    vh_db = float(props['VH'])
                    
                    # Debug: Print first few values
                    if success_count < 3:
                        print(f"    Sample value - VV: {vv_db:.2f} dB, VH: {vh_db:.2f} dB")
                    
                    # S1 GRD data from GEE is already in dB format (negative values)
                    # Just use the values directly, no conversion needed
                    props_clean = {
                        'Site': props['Site'],
                        'Date': None, # Placeholder
                        'Date_Millis': props['Date_Millis'],
                        'VV': vv_db,
                        'VH': vh_db
                    }
                    all_data.append(props_clean)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count <= 3:
                        print(f"    Error processing: {e}")
            else:
                null_count += 1
        
        print(f"  Stats - Success: {success_count}, Null: {null_count}, Negative/Zero: {negative_count}, Errors: {error_count}")

    if all_data:
        df = pd.DataFrame(all_data)
        # Convert Millis to Date
        df['Date'] = pd.to_datetime(df['Date_Millis'], unit='ms').dt.normalize()
        
        # Drop temp column
        df = df.drop(columns=['Date_Millis'])
        
        # Average Daily (Asc/Desc)
        df_daily = df.groupby(['Site', 'Date'])[['VV', 'VH']].mean().reset_index()
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_daily.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! Saved {len(df_daily)} rows.")
        print(df_daily.head())
    else:
        print("No valid data found.")

if __name__ == "__main__":
    download_s1_data()