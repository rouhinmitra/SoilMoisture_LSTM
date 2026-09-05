import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# --- CONFIG ---
INPUT_FILE = "../Data/s2_pixels/presto_embeddings_fused.csv"

def verify():
    print("Loading embeddings...")
    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Select just the embedding columns (0 to 127)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    X = df[emb_cols].values
    
    print(f"Loaded {len(df)} rows with {len(emb_cols)} dimensions.")

    # --- CHECK 1: The "Pulse" (Standard Deviation) ---
    # If std is 0, the neuron is "dead". We want to see variance.
    std_devs = df[emb_cols].std()
    dead_neurons = sum(std_devs < 0.001)
    print(f"\nCHECK 1: NEURON HEALTH")
    print(f"Dead Neurons (Zero Variance): {dead_neurons} / 128")
    if dead_neurons > 64:
        print("❌ WARNING: Over half your embeddings are dead. Something is wrong.")
    else:
        print("✅ Pulse detected. The model is reacting to the data.")

    # --- CHECK 2: Seasonality (Visual Inspection) ---
    # We plot the 3 dimensions with the highest variance over time
    # If these look like random noise, that's bad. They should look like waves.
    top_vars = std_devs.nlargest(3).index.tolist()
    
    plt.figure(figsize=(12, 5))
    for col in top_vars:
        # Sort by date for clean plotting
        site_1 = df[df['Site'] == df['Site'].unique()[0]].sort_values('Date')
        plt.plot(site_1['Date'], site_1[col], label=col, alpha=0.7)
    
    plt.title(f"Time Series of Top 3 Embeddings (Site: {df['Site'].unique()[0]})")
    plt.ylabel("Embedding Value")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print("\nCHECK 2: Look at the plot above.")
    print("  - GOOD: Smooth lines, wave-like patterns (seasonality).")
    print("  - BAD: Jagged 'white noise' or flat lines.")

    # --- CHECK 3: Structure (PCA Projection) ---
    # We project 128D -> 2D and color by Month.
    # We expect distinct clusters for seasons (e.g. Winter points grouped together).
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    df['Month'] = df['Date'].dt.month

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Month'], cmap='RdYlBu', alpha=0.6, s=15)
    plt.colorbar(scatter, label='Month (1=Jan, 12=Dec)')
    plt.title("PCA of Embeddings (Colored by Month)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nCHECK 3: Look at the PCA plot.")
    print("  - GOOD: A circular flow or rainbow gradient (Jan close to Feb, far from Jul).")
    print("  - BAD: A random 'confetti' mess with no color structure.")

if __name__ == "__main__":
    verify()