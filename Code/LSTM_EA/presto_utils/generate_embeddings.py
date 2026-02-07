import torch
import pandas as pd
import numpy as np
import os
import sys
from presto import Presto
from presto import construct_single_presto_input

# --- Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

INPUT_FILE = "../Data/s2_pixels/s2_presto_inputs.csv"
OUTPUT_FILE = "../Data/s2_pixels/presto_embeddings.csv"

# Processing Frame-by-Frame (1 step at a time) to prevent pooling
CHUNK_SIZE = 1

BAND_MAPPING = {
    'B1': 'B1', 'B2': 'B2', 'B3': 'B3', 'B4': 'B4',
    'B5': 'B5', 'B6': 'B6', 'B7': 'B7', 
    'B8': 'B8', 'B8A': 'B8A', 'B9': 'B9', 
    'B11': 'B11', 'B12': 'B12'
}

def clean_and_group_data(df):
    print(f"Raw Data Size: {len(df)} rows")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    numeric_cols = list(BAND_MAPPING.keys()) + ["latitude", "longitude"]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    # Aggregate duplicates (same day)
    grouped_df = df.groupby(["Site", "Date"])[available_cols].mean().reset_index()
    grouped_df["Date"] = pd.to_datetime(grouped_df["Date"])
    
    print(f"Cleaned Data Size: {len(grouped_df)} rows")
    return grouped_df

def generate_embeddings():
    print("Loading Presto Model...")
    
    model_path = "presto_utils/default_model.pt"
    if not os.path.exists(model_path):
        model_path = "default_model.pt"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = Presto.construct()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    model.eval()

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return
    
    raw_df = pd.read_csv(INPUT_FILE)
    df = clean_and_group_data(raw_df)
    
    all_results = []

    # 3. Process each site
    for site_name, site_df in df.groupby("Site"):
        print(f"Processing site: {site_name} (Total rows: {len(site_df)})")
        site_df = site_df.sort_values('Date')
        
        available_bands = [col for col in BAND_MAPPING.keys() if col in site_df.columns]
        
        site_embeddings_list = []
        
        # Iterating row by row (Frame-by-Frame extraction)
        # This ensures we get exactly 1 embedding for every 1 date
        for i in range(len(site_df)):
            # Get single row as a DataFrame (to keep columns)
            chunk = site_df.iloc[[i]]
            
            # Prepare Inputs
            s2_data = chunk[available_bands].values
            s2_tensor = torch.tensor(s2_data, dtype=torch.float32)
            print("s2_tensor shape: ", s2_tensor.shape)
            x, mask, dynamic_world = construct_single_presto_input(
                s2=s2_tensor, 
                s2_bands=available_bands
            )
            print("x shape: ", x.shape)
            print("mask shape: ", mask.shape)
            print("dynamic_world shape: ", dynamic_world.shape)
            print("x: ", x)
            print("mask: ", mask)
            print("dynamic_world: ", dynamic_world)
            # Type Casting
            dynamic_world = dynamic_world.long()
            
            # Add Batch Dims [1, 1, Band]
            x = x.unsqueeze(0)
            print("x unsqueezed: ", x.shape)
            mask = mask.unsqueeze(0)
            print("mask unsqueezed: ", mask.shape)
            dynamic_world = dynamic_world.unsqueeze(0)
            print("dynamic_world unsqueezed: ", dynamic_world.shape)
            
            # Static Inputs
            lat = chunk["latitude"].iloc[0]
            lon = chunk["longitude"].iloc[0]
            latlons = torch.tensor([[lat, lon]]).float()
            
            # Month (0-11)
            start_month = chunk["Date"].iloc[0].month - 1

            month = torch.tensor([start_month]).long()

            # Run Inference
            with torch.no_grad():
                enc_out = model.encoder(x, dynamic_world, latlons, mask, month)
            print("enc_out shape: ", enc_out.shape)
            # enc_out is likely [1, 128] now, which is perfect
            chunk_embedding = enc_out.reshape(1, -1).numpy()
            print("chunk_embedding shape: ", chunk_embedding.shape)
            print("chunk_embedding: ", chunk_embedding)
            site_embeddings_list.append(chunk_embedding)
            
            # Progress bar for sanity
            if i % 50 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(site_df)}")
                sys.stdout.flush()
        
        print(f"\r  Progress: {len(site_df)}/{len(site_df)} - Done")

        # Concatenate all single-step embeddings
        full_embeddings = np.concatenate(site_embeddings_list, axis=0)
        print("full_embeddings shape: ", full_embeddings.shape)
        print("full_embeddings: ", full_embeddings)
        # Verify alignment
        if len(full_embeddings) != len(site_df):
            print(f"  CRITICAL ERROR: Shape mismatch! {len(full_embeddings)} vs {len(site_df)}")
            continue

        # Create DataFrame
        cols = [f"emb_{k}" for k in range(full_embeddings.shape[1])]
        site_emb_df = pd.DataFrame(full_embeddings, columns=cols)
        site_emb_df["Date"] = site_df["Date"].values
        site_emb_df["Site"] = site_name
        print("site_emb_df shape: ", site_emb_df.shape)
        print("site_emb_df: ", site_emb_df)
        all_results.append(site_emb_df)

    # 4. Save
    if all_results:
        final_df = pd.concat(all_results)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Saved embeddings to: {OUTPUT_FILE}")
        print(f"Final Shape: {final_df.shape}")
    else:
        print("No embeddings generated.")

if __name__ == "__main__": 
    generate_embeddings()