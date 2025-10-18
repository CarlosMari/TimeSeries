# file: pca_latent_analysis.py

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import warnings
from tqdm import tqdm
from matplotlib.colors import ListedColormap # <-- ADD THIS LINE

# --- Import from your project files ---
from VAE.models.cvae import LSTM_VAE
from config import model_config

def test_lv_fit_details(x_data, t):
    """
    Performs the LV linear regression test.
    (This function can be moved to a utils.py file and imported)
    """
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

# --- Analysis Parameters ---
# How many samples from the training set to use for the PCA.
# Using all samples can be memory-intensive. -1 for all.
NUM_SAMPLES_FOR_PCA = 20000
BATCH_SIZE = 256 # For encoding the data

# --- Advanced Visualization ---
# WARNING: This is slow as it decodes and tests every point.
# Set to False for a quick overview.
COLOR_PLOT_BY_VALIDITY = True
VALIDITY_THRESHOLD = 0.95
t_time_array = np.arange(65)


# --- MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    # --- Step A: Load Model and Real Data ---
    print("--- Loading Model and Real Dataset ---")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(f"Real data file not found: {REAL_DATA_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTM_VAE(model_config)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    import pickle
    with open(REAL_DATA_PATH, 'rb') as f:
        real_data_dict = pickle.load(f)
    
    data_cube = torch.from_numpy(real_data_dict['data']).float()
    
    # Limit the number of samples if configured
    if NUM_SAMPLES_FOR_PCA != -1 and NUM_SAMPLES_FOR_PCA < len(data_cube):
        data_cube = data_cube[:NUM_SAMPLES_FOR_PCA]
    print(f"Using {len(data_cube)} samples for PCA.")

    dataset = TensorDataset(data_cube)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Step B: Encode Data to get Latent Vectors ---
    print("\n--- Encoding dataset to get latent vectors (mu) ---")
    all_mu_vectors = []
    with torch.no_grad():
        for (batch_X,) in tqdm(dataloader, desc="Encoding data"):
            batch_X = batch_X.to(device)
            # The model forward pass returns (X_hat, mu, log_var, max_vals)
            _, mu, _, _ = model(batch_X)
            all_mu_vectors.append(mu.cpu())
    
    mu_vectors_tensor = torch.cat(all_mu_vectors, dim=0)
    mu_vectors_np = mu_vectors_tensor.numpy()

    # --- Step C: Perform and Analyze PCA ---
    print("\n--- Performing PCA on the latent vectors ---")
    pca = PCA(n_components=model_config['latent_dim'])
    # fit_transform learns the components and transforms the data in one step
    latent_pca = pca.fit_transform(mu_vectors_np)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance of first 3 components: "
          f"PC1={explained_variance[0]:.2%}, "
          f"PC2={explained_variance[1]:.2%}, "
          f"PC3={explained_variance[2]:.2%}")
    print(f"Cumulative variance of first 3 components: {np.sum(explained_variance[:3]):.2%}")

    # --- Step D: Create Scree Plot ---
    plt.figure(figsize=(12, 6))
    plt.bar(range(model_config['latent_dim']), explained_variance, alpha=0.8, align='center',
            label='Individual explained variance')
    plt.step(range(model_config['latent_dim']), np.cumsum(explained_variance), where='mid',
             label='Cumulative explained variance', color='red')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.title('Scree Plot for Latent Space PCA')
    plt.legend(loc='best')
    plt.xticks(range(model_config['latent_dim']))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'./PCA/ScreePlot.png')
    #plt.show()

    # --- Step E: Create Scatter Plot of Principal Components ---
    print("\n--- Generating PCA scatter plot ---")
    validity_colors = None
    if COLOR_PLOT_BY_VALIDITY:
        print("(This will be slow) Checking LV validity for each point to add color...")
        validities = []
        # We decode the mu vectors directly to see what the "mean" prediction is
        for mu_vec in tqdm(mu_vectors_tensor, desc="Decoding for validity"):
            mu_vec = mu_vec.unsqueeze(0).to(device) # Add batch dimension
            norm_seq, pred_max_vals = model.decode(mu_vec)
            unscaled_seq = norm_seq.cpu().numpy()[0].T * pred_max_vals.cpu().numpy()[0]
            fit_details = test_lv_fit_details(unscaled_seq, t_time_array)
            avg_r2 = np.mean(fit_details['r2_scores'])
            validities.append(1 if avg_r2 > VALIDITY_THRESHOLD else 0)
        validity_colors = np.array(validities)
        
    plt.figure(figsize=(12, 10))
    cmap = ListedColormap(['#2244aa', '#44aa22']) if COLOR_PLOT_BY_VALIDITY else 'viridis'
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                          c=validity_colors, 
                          cmap=cmap,
                          alpha=0.6, s=15)
    
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    title = 'Latent Space Projected onto First Two Principal Components'
    if COLOR_PLOT_BY_VALIDITY:
        title += '\n(Colored by LV Validity)'
        # Create a legend for the colors
        handles = [plt.Line2D([0], [0], marker='o', color='w', label='Invalid (R² <= 0.9)', markerfacecolor=cmap(0), markersize=10),
                   plt.Line2D([0], [0], marker='o', color='w', label='Valid (R² > 0.9)', markerfacecolor=cmap(1), markersize=10)]
        plt.legend(handles=handles, title="LV Validity")

    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', alpha=0.3, linewidth=1)
    plt.axvline(0, color='black', alpha=0.3, linewidth=1)
    #plt.show()
    plt.savefig(f'./PCA/2_components.png')

