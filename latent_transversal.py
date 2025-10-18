# file: latent_traversal.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import pickle
from tqdm import tqdm

# --- Import from your project files ---
# (Update the path based on your project structure if needed)
from VAE.models.cvae import LSTM_VAE 
from config import model_config

model_config['n_curves'] = 7
def test_lv_fit_details(x_data, t):
    """Performs the LV linear regression test."""
    from sklearn.linear_model import LinearRegression
    import warnings
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

def get_or_create_pca_model(model, data_path, pca_model_path):
    """Loads a saved PCA model or creates one if it doesn't exist."""
    if os.path.exists(pca_model_path):
        print(f"Loading existing PCA model from {pca_model_path}")
        with open(pca_model_path, 'rb') as f:
            pca, mean_mu = pickle.load(f)
        return pca, mean_mu

    print("No saved PCA model found. Creating a new one...")
    from torch.utils.data import TensorDataset, DataLoader
    
    with open(data_path, 'rb') as f:
        real_data_dict = pickle.load(f)
    data_cube = torch.from_numpy(real_data_dict['data']).float()
    
    dataset = TensorDataset(data_cube)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    device = next(model.parameters()).device
    
    all_mu_vectors = []
    with torch.no_grad():
        for (batch_X,) in tqdm(dataloader, desc="Encoding data for PCA"):
            _, mu, _, _ = model(batch_X.to(device))
            all_mu_vectors.append(mu.cpu())
    
    mu_vectors_np = torch.cat(all_mu_vectors, dim=0).numpy()
    mean_mu = np.mean(mu_vectors_np, axis=0)
    
    pca = PCA(n_components=model.config['latent_dim'])
    pca.fit(mu_vectors_np)
    
    print(f"Saving new PCA model to {pca_model_path}")
    with open(pca_model_path, 'wb') as f:
        pickle.dump((pca, mean_mu), f)
        
    return pca, mean_mu


# --- CONFIGURATION ---
MODEL_SAVE_ROUTE = './model_ckpts/'
MODEL_NAME = "model_new_arch"
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_ROUTE, f"{MODEL_NAME}.pth")
REAL_DATA_PATH = 'data/TRAIN_PREPROCESSED_DS.pkl'
PCA_MODEL_PATH = 'pca_model.pkl' # Path to save/load the PCA results

# --- Traversal Parameters ---
NUM_PCS_TO_TRAVERSE = 3   # How many of the top PCs to visualize
TRAVERSAL_RANGE = 5       # How far to travel along the axis (in std devs)
TRAVERSAL_STEPS = 41      # Number of points to sample along the axis (odd number is good)
VALIDITY_THRESHOLD = 0.9
t_time_array = np.arange(65)


# --- MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    # --- Step A: Load Model and PCA ---
    print("--- Loading Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_VAE(model_config)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    pca, mean_mu = get_or_create_pca_model(model, REAL_DATA_PATH, PCA_MODEL_PATH)
    mean_mu = torch.from_numpy(mean_mu).float().to(device)
    
    # --- Step B: Loop Through PCs and Traverse ---
    for pc_index in range(NUM_PCS_TO_TRAVERSE):
        print(f"\n--- Traversing Principal Component #{pc_index+1} ---")
        
        # Get the direction vector for this principal component
        pc_vector = torch.from_numpy(pca.components_[pc_index]).float().to(device)
        
        # Create the traversal path: a series of steps along the PC
        traversal_values = np.linspace(-TRAVERSAL_RANGE, TRAVERSAL_RANGE, TRAVERSAL_STEPS)
        
        # Construct the batch of latent vectors for this entire traversal
        # z = mean + s * pc_vector
        latent_vectors_to_decode = mean_mu.unsqueeze(0) + torch.from_numpy(traversal_values).float().view(-1, 1).to(device) @ pc_vector.unsqueeze(0)
        
        # Decode all vectors in one batch
        with torch.no_grad():
            norm_seq, pred_max_vals = model.decode(latent_vectors_to_decode)
        
        # Analyze the results of the traversal
        validities = []
        max_abundances = []
        for i in range(TRAVERSAL_STEPS):
            unscaled_seq = norm_seq[i].cpu().numpy().T * pred_max_vals[i].cpu().numpy()
            fit_details = test_lv_fit_details(unscaled_seq, t_time_array)
            avg_r2 = np.mean(fit_details['r2_scores'])
            validities.append(avg_r2)
            max_abundances.append(np.max(unscaled_seq, axis=0))
        
        validities = np.array(validities)
        max_abundances = np.array(max_abundances)

        # --- Step C: Visualize the Traversal ---
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, 
                                 gridspec_kw={'height_ratios': [3, 1, 3]})
        fig.suptitle(f'Latent Space Traversal along Principal Component {pc_index+1}', fontsize=18)

        # Plot 1: How the max abundance of each species changes
        for species_idx in range(model_config['n_curves']):
            axes[0].plot(traversal_values, max_abundances[:, species_idx], label=f'Species {species_idx+1}')
        axes[0].set_title('Max Abundance of Each Species vs. Traversal')
        axes[0].set_ylabel('Max Abundance')
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[0].grid(True, linestyle='--')

        # Plot 2: How the LV validity (R² score) changes
        axes[1].plot(traversal_values, validities, color='black', marker='.')
        axes[1].axhline(VALIDITY_THRESHOLD, color='red', linestyle='--', label=f'Validity Threshold ({VALIDITY_THRESHOLD})')
        axes[1].set_title('Lotka-Volterra Validity (Average R²) vs. Traversal')
        axes[1].set_ylabel('Avg. R² Score')
        axes[1].set_ylim(bottom=min(0.5, np.min(validities) - 0.1), top=1.05)
        axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[1].grid(True, linestyle='--')

        # Plot 3: Show example time-series plots from the traversal
        example_indices = [0, TRAVERSAL_STEPS // 2, TRAVERSAL_STEPS - 1]
        for i, step_idx in enumerate(example_indices):
            s_val = traversal_values[step_idx]
            unscaled_seq = norm_seq[step_idx].cpu().numpy().T * pred_max_vals[step_idx].cpu().numpy()
            offset = i * 2.5 # Offset curves vertically for clarity
            for species_idx in range(model_config['n_curves']):
                line, = axes[2].plot(t_time_array, unscaled_seq[:, species_idx] + offset)
            axes[2].text(-5, offset + np.max(unscaled_seq)/2, f's = {s_val:.1f}', verticalalignment='center')
        axes[2].set_title('Example Generated Time Series at Different Points Along Traversal')
        axes[2].set_xlabel(f'Position along PC {pc_index+1} (in standard deviations from mean)')
        axes[2].set_ylabel('Abundance (vertically offset)')
        axes[2].set_yticks([]) # Hide y-axis ticks for clarity
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'PCA/transversal_{pc_index}.png')
        plt.show()
