# file: systematic_latent_slicing.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import warnings
from tqdm import tqdm

# --- Import from your project files ---
from VAE.models.cvae import LSTM_VAE
from config import model_config

def test_lv_fit_details(x_data, t):
    """
    Performs the LV linear regression test and returns full details for plotting.
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

def create_latent_slice_plot(model, dim_x, dim_y, grid_res, plot_rng, threshold, t_array):
    """
    Generates and displays a single 2D validity plot for a given pair of dimensions.
    """
    validity_grid = np.zeros((grid_res, grid_res))
    x_coords = np.linspace(-plot_rng, plot_rng, grid_res)
    y_coords = np.linspace(-plot_rng, plot_rng, grid_res)
    
    latent_dim = model.config['latent_dim']
    device = next(model.parameters()).device

    for i, y_val in enumerate(tqdm(y_coords, desc=f"Scanning z{dim_x} vs z{dim_y}")):
        for j, x_val in enumerate(x_coords):
            with torch.no_grad():
                z_sample = torch.zeros(1, latent_dim, device=device)
                z_sample[0, dim_x] = x_val
                z_sample[0, dim_y] = y_val

                norm_seq, pred_max_vals = model.decode(z_sample)

            unscaled_seq = norm_seq.cpu().numpy()[0].T * pred_max_vals.cpu().numpy()[0]
            fit_details = test_lv_fit_details(unscaled_seq, t_array)
            avg_r2 = np.mean(fit_details['r2_scores'])
            
            if avg_r2 > threshold:
                validity_grid[i, j] = avg_r2 #1

    # Visualization
    #cmap = matplotlib.colors #ListedColormap(['#2244aa', '#44aa22'])
    plt.viridis()
    fig, ax = plt.subplots(figsize=(10, 10))
    #fig.colorbar(ax=ax)
    im = ax.imshow(validity_grid, origin='lower',
                   extent=[-plot_rng, plot_rng, plot_rng, -plot_rng],
                   interpolation='hanning')
    for r in [1, 2, 3]:
        circle = plt.Circle((0, 0), r, color='black', fill=False, linestyle='-', alpha=0.5)
        ax.add_artist(circle)
    outer_circle = plt.Circle((0, 0), plot_rng, transform=ax.transData, color='black', fill=False, linewidth=3)
    im.set_clip_path(outer_circle)
    ax.set_xlabel(f"Latent dimension z{dim_x}", fontsize=12)
    ax.set_ylabel(f"Latent dimension z{dim_y}", fontsize=12)
    ax.set_title(f"Latent Space Validity (z{dim_x} vs z{dim_y})", fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.savefig(f'./data_checker/R2_validity_{dim_x}_{dim_y}_extra.png')
    plt.show()

# --- CONFIGURATION ---

# --- Model paths ---
MODEL_SAVE_ROUTE = './model_ckpts/'
MODEL_NAME = "model_new_arch"
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_ROUTE, f"{MODEL_NAME}.pth")

# --- Latent Space Slicing Parameters ---
# !! DEFINE THE PAIRS OF DIMENSIONS YOU WANT TO PLOT HERE !!
# For a 20D space, indices are 0 through 19.
'''(0, 1),      # The standard first two dimensions
    (0, 10),     # How does the first dim interact with one in the middle?
    (2, 3),      # Another adjacent pair
    (18, 19),     # The last two dimensions
    (0,24),'''
PAIRS_TO_PLOT = [
    
    (1,5),
    (1,24),
    (0,5),
    (0,24)
]

# --- Analysis Parameters ---
GRID_RESOLUTION = 50      # Points per axis. Lower for faster checks (e.g., 25).
PLOT_RANGE = 3.5          # How far from the origin to plot.
VALIDITY_THRESHOLD = 0.95  # R² score required to be "valid".
t_time_array = np.arange(65)


# --- MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    # --- Step A: Validate Configuration and Load Model ---
    max_dim_requested = max(max(pair) for pair in PAIRS_TO_PLOT)
    if max_dim_requested >= model_config['latent_dim']:
        raise ValueError(f"Error: You requested to plot dimension {max_dim_requested}, "
                         f"but the model's latent space is only {model_config['latent_dim']} dimensional.")

    print("--- Loading Neural Network Model ---")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTM_VAE(model_config)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)

    # --- Step B: Loop Through Pairs and Generate Plots ---
    for dim_x, dim_y in PAIRS_TO_PLOT:
        print("\n" + "="*60)
        print(f"Generating plot for dimensions z{dim_x} vs z{dim_y}")
        print("="*60)
        
        create_latent_slice_plot(
            model=model,
            dim_x=dim_x,
            dim_y=dim_y,
            grid_res=GRID_RESOLUTION,
            plot_rng=PLOT_RANGE,
            threshold=VALIDITY_THRESHOLD,
            t_array=t_time_array
        )

    print("\nSystematic slicing complete.")
