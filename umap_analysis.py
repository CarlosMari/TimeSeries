# file: umap_latent_analysis.py

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
import pickle

# --- Import from your project files ---
from VAE.models.cvae import LSTM_VAE 
from config import model_config

# --- UMAP and Plotting Imports ---
import umap
import umap.plot

def test_lv_fit_details(x_data, t):
    """Performs the LV linear regression test."""
    from sklearn.linear_model import LinearRegression
    N = x_data.shape[1]
    r2_scores = []
    for i in range(N):
        x_i_positive = np.maximum(x_data[:, i], 1e-9)
        log_xi = np.log(x_i_positive)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d_log_xi_dt = np.gradient(log_xi, t)
        model_lr = LinearRegression()
        model_lr.fit(x_data, d_log_xi_dt)
        r2 = model_lr.score(x_data, d_log_xi_dt)
        r2_scores.append(r2)
    return {'r2_scores': np.array(r2_scores)}

# --- CONFIGURATION ---

# --- Model and Data Paths ---
MODEL_SAVE_ROUTE = './model_ckpts/'
MODEL_NAME = "model_new_arch"
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_ROUTE, f"{MODEL_NAME}.pth")
REAL_DATA_PATH = 'data/TRAIN_PREPROCESSED_DS.pkl'
UMAP_RESULTS_PATH = 'umap_results.pkl' # To save/load computed results

# --- Analysis Parameters ---
NUM_SAMPLES_FOR_UMAP = 5000  # -1 for all. Fewer is faster.
BATCH_SIZE = 256
VALIDITY_THRESHOLD = 0.9
t_time_array = np.arange(65)

# UMAP hyperparameters. These are good starting points.
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'euclidean'


# --- MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    # --- Step A: Load Model and Real Data ---
    print("--- Loading Model and Real Dataset ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_VAE(model_config)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # --- Step B: Get Latent Vectors and Calculate Properties ---
    # Check if we have pre-computed results to save time
    if os.path.exists(UMAP_RESULTS_PATH):
        print(f"Loading pre-computed results from {UMAP_RESULTS_PATH}")
        with open(UMAP_RESULTS_PATH, 'rb') as f:
            results = pickle.load(f)
        mu_vectors_np = results['mu_vectors']
        validities = results['validities']
        dominant_species = results['dominant_species']
    else:
        print("No saved results found. Encoding dataset and calculating properties...")
        with open(REAL_DATA_PATH, 'rb') as f:
            real_data_dict = pickle.load(f)
        data_cube = torch.from_numpy(real_data_dict['data']).float()
        
        if NUM_SAMPLES_FOR_UMAP != -1 and NUM_SAMPLES_FOR_UMAP < len(data_cube):
            data_cube = data_cube[:NUM_SAMPLES_FOR_UMAP]
        print(f"Using {len(data_cube)} samples.")

        dataset = TensorDataset(data_cube)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        all_mu_vectors = []
        validities = []
        dominant_species = []

        with torch.no_grad():
            for (batch_X,) in tqdm(dataloader, desc="Encoding & Analyzing Data"):
                batch_X = batch_X.to(device)
                _, mu, _, _ = model(batch_X)
                all_mu_vectors.append(mu.cpu())
                
                # Decode to check properties. This is the slow part.
                norm_seq, pred_max_vals = model.decode(mu)
                
                for i in range(len(norm_seq)):
                    unscaled_seq = norm_seq[i].cpu().numpy().T * pred_max_vals[i].cpu().numpy()
                    fit_details = test_lv_fit_details(unscaled_seq, t_time_array)
                    avg_r2 = np.mean(fit_details['r2_scores'])
                    validities.append(1 if avg_r2 > VALIDITY_THRESHOLD else 0)
                    dominant_species.append(np.argmax(np.max(unscaled_seq, axis=0)))

        mu_vectors_np = torch.cat(all_mu_vectors, dim=0).numpy()
        validities = np.array(validities)
        dominant_species = np.array(dominant_species)
        
        print(f"Saving computed results to {UMAP_RESULTS_PATH}")
        with open(UMAP_RESULTS_PATH, 'wb') as f:
            pickle.dump({
                'mu_vectors': mu_vectors_np,
                'validities': validities,
                'dominant_species': dominant_species
            }, f)

    # --- Step C: Compute UMAP Embedding ---
    print("\n--- Computing UMAP embedding ---")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        n_components=2,
        random_state=42, # for reproducibility
        verbose=True
    )
    embedding = reducer.fit_transform(mu_vectors_np)
    print("UMAP computation complete.")

    # --- Step D: Visualize the Embeddings ---
    
    # Plot 1: Colored by LV Validity
    print("Generating plot colored by LV Validity...")
    fig, ax = plt.subplots(figsize=(12, 10))
    # Use 'datashade' for large datasets if installed, otherwise use scatter
        # Plot 1: Colored by LV Validity
    print("Generating plot colored by LV Validity...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    try:
        # ---- ROBUST METHOD USING PANDAS DATAFRAME ----
        import datashader as ds
        from datashader.mpl_ext import dsshow
        from matplotlib.colors import ListedColormap
        import pandas as pd

        # 1. Create a pandas DataFrame from your data
        df = pd.DataFrame(embedding, columns=['x', 'y'])
        
        # 2. Add the validity data as a categorical column
        df['validity'] = pd.Series(validities).astype('category')

        # 3. Define the color key
        color_key = {0: '#2244aa', 1: '#44aa22'} # 0: Invalid, 1: Valid

        # 4. Use dsshow with the count_cat aggregator
        # This is the correct, modern way to do categorical aggregation
        dsshow(
            data=df,
            glyph=ds.Point,
            x='x',
            y='y',
            agg=ds.count_cat('validity'), # Use the name of the categorical column
            color_key=color_key,
            ax=ax,
            shade_hook=ds.tf.set_background('black')
        )
        
        # Create a legend manually for the datashader plot
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label='Invalid (R² <= 0.9)', markerfacecolor=color_key[0], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Valid (R² > 0.9)', markerfacecolor=color_key[1], markersize=10)
        ]
        plt.legend(handles=handles, title="LV Validity")

    except ImportError:
        # Fallback to standard scatter plot if datashader is not available
        print("Warning: datashader not found. Using matplotlib.scatter which may be slow.")
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#2244aa', '#44aa22'])
        plt.scatter(embedding[:, 0], embedding[:, 1], c=validities, cmap=cmap, s=5, alpha=0.7)
        # Create a legend for the scatter plot
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label='Invalid (R² <= 0.9)', markerfacecolor=cmap(0), markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Valid (R² > 0.9)', markerfacecolor=cmap(1), markersize=10)
        ]
        plt.legend(handles=handles, title="LV Validity")


    ax.set_title('UMAP Projection of Latent Space, Colored by LV Validity', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.show()

    plt.savefig('./UMAP/validity.png')

    # Plot 2: Colored by Dominant Species
    print("Generating plot colored by Dominant Species...")
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=dominant_species, cmap='jet', s=5, alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Species")
    ax.add_artist(legend1)
    ax.set_title('UMAP Projection of Latent Space, Colored by Dominant Species', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.show()
    plt.savefig('./UMAP/dominant.png')

