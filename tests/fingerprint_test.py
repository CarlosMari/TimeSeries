import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap

# --- Configuration ---
PROCESSED_DATA_FILE = 'data/TRAIN_PREPROCESSED.pkl'
NUM_EXAMPLES_TO_PLOT = 4 # How many top examples to show for each class

# --- Define the ideal target fingerprints we are looking for ---
# These are the same targets we use in the generation function.
# Format: [Activity, Periodicity, Transience]
TARGET_FINGERPRINTS = {
    "Stable": np.array([0.0, 0.0, 0.0]),
    "Oscillating": np.array([1.0, 1.0, 0.0]),
    "Capacitor-Like": np.array([0.5, 0.0, 1.0]),
}

# --- Main Script ---
if __name__ == "__main__":
    print(f"Loading pre-processed data from: {PROCESSED_DATA_FILE}")
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data_package = pickle.load(f)

    curves = data_package['data']
    fingerprints = data_package['labels']
    
    print(f"Loaded {len(curves)} samples.")
    
    # --- Create a single large plot ---
    fig, axes = plt.subplots(len(TARGET_FINGERPRINTS), NUM_EXAMPLES_TO_PLOT, 
                             figsize=(18, 12), constrained_layout=True)
    
    fig.suptitle("Best Examples from Dataset Matching Target Fingerprints", fontsize=20)

    # --- Find and plot the best examples for each class ---
    for row, (class_name, target_fp) in enumerate(TARGET_FINGERPRINTS.items()):
        
        # Calculate the Euclidean distance from every sample's fingerprint to the target
        distances = np.linalg.norm(fingerprints - target_fp, axis=1)
        
        # Get the indices of the samples with the smallest distances
        best_indices = np.argsort(distances)[:NUM_EXAMPLES_TO_PLOT]
        
        print(f"\n--- Top examples for class: {class_name} ---")
        print(f"Target Fingerprint: {target_fp}")

        # Plot each of the best examples
        for col, idx in enumerate(best_indices):
            ax = axes[row, col]
            curves_to_plot = curves[idx]
            actual_fp = fingerprints[idx]
            
            # Print the info for debugging
            print(f"  - Index {idx}: Distance={distances[idx]:.3f}, Actual FP={np.round(actual_fp, 2)}")

            # Plot the 7 curves in the family
            for i in range(curves_to_plot.shape[0]):
                ax.plot(curves_to_plot[i])
            
            # Set titles and labels
            if col == 0:
                ax.set_ylabel(class_name, fontsize=14, weight='bold')
            ax.set_title(f"Index {idx} (Dist: {distances[idx]:.2f})")
            ax.set_ylim([-0.1, 1.1]) # Use the normalized data range

    plt.savefig("fingerprint_verification.png")
    print("\nPlot saved as fingerprint_verification.png")
    plt.show()