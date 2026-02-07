import requests
import os

# --- PERMANENT LINK (Frozen Commit Hash) ---
# This points to the file as it existed in commit 5afde40
url = "https://github.com/nasaharvest/presto/raw/5afde40850d73bfaed26078fc3bda621a55c311d/data/default_model.pt"

output_path = "presto_utils/default_model.pt"

print(f"Downloading model from {url}...")
print("This may take a few minutes (approx 270 MB)...")

try:
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for 404 errors

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    # Verify file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSuccess! Saved to {output_path}")
    print(f"File size: {size_mb:.2f} MB")
    
    if size_mb < 100:
        print("WARNING: The file is too small. It might be a Git LFS pointer text file.")
        print("If generate_embeddings.py fails, try downloading manually from the URL above.")

except Exception as e:
    print(f"\nFailed: {e}")