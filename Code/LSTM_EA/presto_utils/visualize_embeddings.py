#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
def visualize_embeddings(embeddings):
    print("Visualizing embeddings...")
    print("embeddings shape: ", embeddings.shape)
    print("embeddings: ", embeddings)
    plt.figure(figsize=(10, 10))
    sns.heatmap(embeddings, annot=True, cmap='viridis')
    plt.show()
def timeseries_plot(df, site, col, year):
    print("Plotting timeseries for site: ", site)
    print("Column: ", col)
    fig, ax = plt.subplots(figsize=(10, 5))
    df = df[(df["Site"] == site) & (df["Date"].dt.year == year)]
    ax.plot(df["Date"], df[col],'o-', label=col, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Embedding")
    ax.set_title("Embedding Timeseries")
    ax.legend()
    plt.show()

#%%
if __name__ == "__main__":
    embeddings = pd.read_csv("/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/presto_embeddings_interpolated.csv")
    embeddings["Date"] = pd.to_datetime(embeddings["Date"])
    timeseries_plot(embeddings, "US-Ne1", "emb_31", 2018)
# %%
