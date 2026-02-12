import torch
import pandas as pd
import numpy as np
import os
import sys
from presto import Presto
from presto import construct_single_presto_input

# --- Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

S2_INPUT_FILE = "../Data/s2_pixels/s2_presto_inputs.csv"
S1_INPUT_FILE = "../Data/s1_pixels/s1_presto_inputs.csv"
OUTPUT_FILE = "../Data/s2_pixels/presto_embeddings_fused.csv"
INTERPOLATED_OUTPUT_FILE = "../Data/s2_pixels/presto_embeddings_fused_interpolated.csv"

# Band Mappings
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S1_BANDS = ['VV', 'VH']

def load_and_prep_data():
    """Loads S1 and S2, normalizes dates to midnight, and merges."""
    print("Loading Data...")
    
    # --- 1. Load S2 (Optical) ---
    if os.path.exists(S2_INPUT_FILE):
        s2_df = pd.read_csv(S2_INPUT_FILE)
        
        # FIX: Normalize to Midnight immediately
        s2_df["Date"] = pd.to_datetime(s2_df["Date"]).dt.normalize()
        
        # Group by day (averaging if multiple satellite strips cover the same site same day)
        s2_agg_cols = S2_BANDS + ["latitude", "longitude"]
        s2_df = s2_df.groupby(["Site", "Date"])[s2_agg_cols].mean().reset_index()
        print(f"  Loaded S2: {len(s2_df)} rows (Daily Aggregated)")
    else:
        print("CRITICAL: S2 File not found.")
        return None

    # --- 2. Load S1 (Radar) ---
    if os.path.exists(S1_INPUT_FILE):
        s1_df = pd.read_csv(S1_INPUT_FILE)
        
        # FIX: Normalize to Midnight immediately
        s1_df["Date"] = pd.to_datetime(s1_df["Date"]).dt.normalize()
        
        # Group by day
        s1_df = s1_df.groupby(["Site", "Date"])[S1_BANDS].mean().reset_index()
        print(f"  Loaded S1: {len(s1_df)} rows (Daily Aggregated)")
    else:
        print("CRITICAL: S1 File not found.")
        return None

    # --- 3. Merge (Outer Join) ---
    # Now that both are 00:00:00, they will match perfectly
    print("Merging S1 and S2 timelines...")
    merged_df = pd.merge(s2_df, s1_df, on=["Site", "Date"], how="outer")

    # --- 4. Fix Coordinates for S1-only days ---
    merged_df['latitude'] = merged_df.groupby('Site')['latitude'].transform(lambda x: x.ffill().bfill())
    merged_df['longitude'] = merged_df.groupby('Site')['longitude'].transform(lambda x: x.ffill().bfill())

    merged_df = merged_df.dropna(subset=['latitude', 'longitude'])
    
    # Sort by date to be clean
    merged_df = merged_df.sort_values(['Site', 'Date'])
    
    print(f"  Final Fused Timeline: {len(merged_df)} rows")
    return merged_df

def generate_embeddings():
    print("Loading Presto Model...")
    model_path = "presto_utils/default_model.pt"
    if not os.path.exists(model_path):
        model_path = "default_model.pt"

    model = Presto.construct()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    model.eval()

    # Load the merged data
    df = load_and_prep_data()
    if df is None: return

    all_results = []

    # Process per Site
    for site_name, site_df in df.groupby("Site"):
        print(f"Processing site: {site_name} (Total rows: {len(site_df)})")
        
        site_embeddings_list = []
        
        for i in range(len(site_df)):
            row = site_df.iloc[[i]]
            
            # --- 1. Prepare S2 Input ---
            if pd.notna(row['B1'].values[0]):
                s2_data = row[S2_BANDS].values
                s2_tensor = torch.tensor(s2_data, dtype=torch.float32)
                s2_input_bands = S2_BANDS
            else:
                s2_tensor = None
                s2_input_bands = None

            # --- 2. Prepare S1 Input ---
            if pd.notna(row['VV'].values[0]):
                s1_data = row[S1_BANDS].values
                s1_tensor = torch.tensor(s1_data, dtype=torch.float32)
                s1_input_bands = S1_BANDS
            else:
                s1_tensor = None
                s1_input_bands = None

            # --- 3. Construct Presto Input ---
            x, mask, dynamic_world = construct_single_presto_input(
                s2=s2_tensor, s2_bands=s2_input_bands,
                s1=s1_tensor, s1_bands=s1_input_bands
            )
            
            dynamic_world = dynamic_world.long().unsqueeze(0)
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            lat = row["latitude"].iloc[0]
            lon = row["longitude"].iloc[0]
            latlons = torch.tensor([[lat, lon]]).float()
            
            month_idx = row["Date"].iloc[0].month - 1
            month = torch.tensor([month_idx]).long()

            with torch.no_grad():
                enc_out = model.encoder(x, dynamic_world, latlons, mask, month)
            
            site_embeddings_list.append(enc_out.reshape(1, -1).numpy())

            if i % 100 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(site_df)}")
                sys.stdout.flush()

        print(f"\r  Progress: {len(site_df)}/{len(site_df)} - Done")

        full_embeddings = np.concatenate(site_embeddings_list, axis=0)
        cols = [f"emb_{k}" for k in range(full_embeddings.shape[1])]
        site_emb_df = pd.DataFrame(full_embeddings, columns=cols)
        site_emb_df["Date"] = site_df["Date"].values
        site_emb_df["Site"] = site_name
        
        all_results.append(site_emb_df)

    if all_results:
        final_df = pd.concat(all_results)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Fused Embeddings saved to: {OUTPUT_FILE}")
        return final_df
    else:
        print("No embeddings generated.")
        return None

def interpolate_embeddings(df):
    """Interpolating embeddings to daily values per site and year"""
    if df is None: return
    
    print("\nInterpolating gaps...")
    df["Date"] = pd.to_datetime(df["Date"])
    interpolated_chunks = []

    for site in df["Site"].unique():
        site_df = df[df["Site"] == site]
        for year in site_df["Date"].dt.year.unique():
            daily_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
            daily_df = pd.DataFrame({"Date": daily_dates})
            
            merged_df = pd.merge(daily_df, site_df, on="Date", how="left")
            merged_df["Site"] = site
            
            numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
            merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear', limit_direction='both')
            
            interpolated_chunks.append(merged_df)

    final_df = pd.concat(interpolated_chunks, ignore_index=True)
    final_df.to_csv(INTERPOLATED_OUTPUT_FILE, index=False)
    print(f"Success! Interpolated data saved to: {INTERPOLATED_OUTPUT_FILE}")
    return final_df

if __name__ == "__main__": 
    embeddings = generate_embeddings()
    interpolated_embeddings = interpolate_embeddings(embeddings)
    if interpolated_embeddings is not None:
        print("Final Shape:", interpolated_embeddings.shape)