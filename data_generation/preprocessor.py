import numpy as np
import pickle

# --- Configuration ---
# Change these paths for your TRAIN and TEST sets
INPUT_FILE_PATH = 'data/TEST_ORDERED_DOWNSAMPLED.pkl'  # <-- Your original data
OUTPUT_FILE_PATH = 'data/TEST_PREPROCESSED_DS.pkl' # <-- The final file for the training loop

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting Pre-processing for {INPUT_FILE_PATH} ---")

    # 1. Load raw data
    with open(INPUT_FILE_PATH, 'rb') as file:
        raw_data = pickle.load(file)
    print(f"Loaded raw data with shape: {raw_data.shape}")

    # --- NEW STEP 1.A: Normalize each family by its absolute max value ---
    # Find the max value across all 7 curves for each sample (N).
    # The axes=(1, 2) finds the max over the 7 curves and 65 time points.
    # keepdims=True makes the shape (N, 1, 1) for easy broadcasting.
    family_max_values = np.max(raw_data, axis=(1, 2), keepdims=True)
    
    # Avoid division by zero for any samples that are all zeros.
    family_max_values[family_max_values == 0] = 1e-8
    
    # Divide each family of curves by its single max value.
    # The result is that the highest peak in each family is now 1.0.
    family_normalized_data = raw_data / family_max_values
    print("Step 1: Each sample family normalized by its global maximum value.")

    # --- Step 2: Sort curves within the NOW-NORMALIZED data ---
    # This ensures a consistent ordering of curves for the model.
    # Note: We now operate on `family_normalized_data` for all subsequent steps.
    max_values_for_sorting = np.max(family_normalized_data, axis=2)
    sorted_indices = np.argsort(-max_values_for_sorting, axis=1)
    data_sorted = np.take_along_axis(family_normalized_data, sorted_indices[:, :, np.newaxis], axis=1)
    print("Step 2: Curves sorted by peak value within each normalized sample.")

    # --- Step 3: Normalize each curve individually to [0, 1] ---
    # This step now makes the peak of *each individual curve* equal to 1.
    # We find the max of each individual curve (from the already-scaled data).
    max_values_per_curve = np.max(data_sorted, axis=2, keepdims=True)

    # Avoid division by zero for curves that might be all zeros
    max_values_per_curve[max_values_per_curve == 0] = 1e-8

    # Normalize each curve by its own max value
    final_normalized_curves = data_sorted / max_values_per_curve
    print("Step 3: Each curve normalized individually to the range [0, 1].")

    # --- Step 4: Prepare and save the final data package ---
    # This package now contains all information needed for full reconstruction.
    final_data_package = {
        # The data for the VAE (each curve's max is 1.0)
        'data': final_normalized_curves,                                     # Shape: (N, 7, 65)
        
        # The max of each curve *relative to its family's max*
        'reconstruction_max_values': np.squeeze(max_values_per_curve, axis=2),# Shape: (N, 7)
        
        # The original max value of each family
        'family_max_values': np.squeeze(family_max_values, axis=(1,2))        # Shape: (N,)
    }
    
    with open(OUTPUT_FILE_PATH, 'wb') as dumpfile:
        pickle.dump(final_data_package, dumpfile)
    
    print(f"\n--- Pre-processing complete! Saved final data package to: {OUTPUT_FILE_PATH} ---")
    print(f"Final data shape: {final_data_package['data'].shape}")
    print(f"Saved per-curve max values shape: {final_data_package['reconstruction_max_values'].shape}")
    print(f"Saved per-family max values shape: {final_data_package['family_max_values'].shape}")