import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from VAE.models.CH_VAE import CHVAE
from config import model_config
import pickle


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualize_vae_filters(model, sample_data, top_n=10):
    """
    Visualize the most relevant filters and their activations in a VAE model.
    
    Args:
        model: Your CHVAE model
        sample_data: Sample time series data (batch_size, in_channels, seq_length)
        top_n: Number of top filters to visualize
    """

    # Make sure model is in eval mode
    model.eval()
    
    # Dictionary to store activations
    activations = {}
    
    # Sacamos las activations = outputs de cada capa
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    
    # Register hooks for the first convolutional layer
    target_layer = "encoder.4"
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(get_activation(target_layer))
    
    # Forward pass through the model
    with torch.no_grad():
        model.encoder(sample_data)
    
    # Get the weights and activations
    for name, module in model.named_modules():
        if name == target_layer:
            weights = module.weight.data.cpu().numpy()
            break
    
    # Get feature maps (activations)
    feature_maps = activations[target_layer].squeeze().cpu().numpy()
    
    # For a batch, take the average across the batch
    if len(feature_maps.shape) == 3:
        feature_maps = feature_maps.mean(axis=0)
    
    # Calculate filter importance based on activation magnitude
    filter_importance = np.mean(np.abs(feature_maps), axis=1)
    
    # Get indices of top N filters
    top_filter_indices = np.argsort(filter_importance)[-top_n:][::-1]
    
    # Plotting
    fig = plt.figure(figsize=(15, 3 * top_n))

    for i, filter_idx in enumerate(top_filter_indices):
        # 1. Plot the filter weights with peaks highlighted
        plt.subplot(top_n, 2, i * 2 + 1)  # 2 columns per row
        plt.plot(weights[filter_idx, 0, :], label='Weights')  # Assuming 1 input channel
        plt.title(f'Filter {filter_idx} Weights')
        plt.xlabel('Índice')

        # Highlight key peaks
        filter_data = weights[filter_idx, 0, :]
        abs_filter = np.abs(filter_data)
        peaks, _ = find_peaks(abs_filter, prominence=np.mean(abs_filter) / 2)
        plt.plot(peaks, filter_data[peaks], "x", color='red', markersize=8, label='Key Components')

        plt.legend()

        # 2. Plot the filter's activation
        plt.subplot(top_n, 2, i * 2 + 2)
        plt.plot(feature_maps[filter_idx, :], label='Activation', color='blue')
        plt.title(f'Filter {filter_idx} Output')
        plt.xlabel('Time Step')

        plt.legend()
    
    plt.tight_layout()
    print('Saving vaefilters')
    plt.savefig('./filters/vaefilters.png')
    plt.show()
    
    return {
        'top_filters': top_filter_indices,
        'filter_importance': filter_importance[top_filter_indices]
    }

def visualize_latent_activations(model, sample_data):
    """
    Visualize how the latent space is activated by the sample data.
    
    Args:
        model: Your CHVAE model
        sample_data: Sample time series data (batch_size, in_channels, seq_length)
    """
    # Make sure model is in eval mode
    model.eval()
    
    # Forward pass up to the latent space
    with torch.no_grad():
        encoded = model.encoder(sample_data)
        batch_size = encoded.size(0)
        encoded_flat = encoded.view(batch_size, -1)
        
        # Get to the latent representations
        corr_output = model.corr(encoded_flat)
        means = model.mean_map(corr_output)
        log_vars = model.std_map(corr_output)
        
        # No need to sample, just use the means for visualization
        z = means
    
    # Convert to numpy for plotting
    z_np = z.cpu().numpy()
    
    # Create a plot
    plt.figure(figsize=(12, 8))
    
    # For each dimension in latent space
    latent_dim = z_np.shape[1]
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(latent_dim)))
    
    for i in range(latent_dim):
        plt.subplot(grid_size, grid_size, i+1)
        
        # For each sample, plot its value in this latent dimension
        plt.hist(z_np[:, i], bins=20, alpha=0.7)
        plt.title(f'Latent Dim {i+1}')
        plt.axvline(x=np.mean(z_np[:, i]), color='r', linestyle='--')
        
    plt.tight_layout()
    plt.suptitle("Distribution of Latent Space Activations", y=1.02)
    plt.savefig('filters/LatentDist.png')
    plt.show()

def time_domain_filter_response(model, sample_data, filter_indices, window_length=129):
    """
    Visualize how specific filters respond to different parts of the time series.
    
    Args:
        model: Your CHVAE model
        sample_data: Sample time series data (single sample: in_channels, seq_length)
        filter_indices: List of filter indices to visualize
        window_length: Length of the sliding window
    """
    # Extract a single time series (first channel, first batch)
    if len(sample_data.shape) == 3:  # If batch is included
        time_series = sample_data[0, 0].cpu().numpy()
    else:
        time_series = sample_data[0].cpu().numpy()
    
    # Make sure model is in eval mode
    model.eval()
    
    # Get the weights of the first conv layer
    for name, module in model.named_modules():
        if name == "encoder.0":
            weights = module.weight.data.cpu().numpy()
            break
    
    # Create the figure
    plt.figure(figsize=(15, 5 * len(filter_indices)))
    
    # Plot original time series
    plt.subplot(len(filter_indices) + 1, 1, 1)
    plt.plot(time_series)
    plt.title('Original Time Series')
    
    # For each filter
    for i, filter_idx in enumerate(filter_indices):
        # Get the filter weights
        filter_weights = weights[filter_idx, 0, :]
        
        # Hacemos la convolución de manera manual
        response = np.zeros_like(time_series)
        pad_size = 2
        padded_series = np.pad(time_series, (pad_size, pad_size), 'constant')
        
        for t in range(len(time_series)):
            window = padded_series[t:t + len(filter_weights)]
            response[t] = np.sum(window * filter_weights)
        
        # Normalize response for visualization
        response = response / np.max(np.abs(response))
        
        # Plot the filter response
        plt.subplot(len(filter_indices) + 1, 1, i + 2)
        plt.plot(response)
        plt.title(f'Filter {filter_idx} Response')
        
        # Highlight areas of high activation
        threshold = 0.5
        high_activation = np.where(np.abs(response) > threshold)[0]
        if len(high_activation) > 0:
            plt.vlines(high_activation, -1, 1, colors='r', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('filters/TimeResponses.png')
    plt.show()


if __name__ == '__main__':
    # Set device and load data
    
    with open('data/TEST_NEW_DIST.pkl', 'rb') as file:
        X = pickle.load(file)
    
    # Normalize data
    X = (X - X.min())/(X.max() - X.min())
    X = torch.from_numpy(X).to(torch.float32)
    
    
    # Load model
    model = CHVAE(model_config).to(DEVICE)
    model.load_state_dict(torch.load(f'{model_config['save_route']}{model_config['name']}.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    
    # Prepare data for visualization
    # Reshape data if needed (needs to be [batch, channels, sequence_length])
    if len(X.shape) == 2:  # If data is [samples, sequence_length]
        X = X.unsqueeze(1)  # Add channel dimension -> [samples, 1, sequence_length]
    
    # Select a small batch for visualization 
    sample_batch = X[23:63].to(DEVICE)
    
    print(f"Input data shape: {sample_batch.shape}")
    
    # 1. Sacamos los outputs de las convoluciones de la primera capa
    top_filter_info = visualize_vae_filters(model, sample_batch, top_n=5)
    print(f"Top filters by activation: {top_filter_info['top_filters']}")
    print(f"Importance scores: {top_filter_info['filter_importance']}")
    
    # 2. Visualizar dimensiones de la capa latente
    visualize_latent_activations(model, sample_batch)
    
    # 3. Ver las salidas para cada output
    single_example = sample_batch[0:1]
    
    # Visualize response of top 3 filters
    time_domain_filter_response(model, single_example, top_filter_info['top_filters'][:5])
    
    # 4. Additional: Visualize reconstructions to see what information is preserved
    with torch.no_grad():
        # Forward pass
        encoded = model.encoder(single_example)
        batch_size = encoded.size(0)
        encoded_flat = encoded.view(batch_size, -1)
        
        # Get latent representation
        corr_output = model.corr(encoded_flat)
        means = model.mean_map(corr_output)
        log_vars = model.std_map(corr_output)
        
        # For visualization, just use the mean (no sampling)
        z = means
        
        # Decode
        decoded_flat = model.linear2(z)
        decoded_reshaped = decoded_flat.view(batch_size, model.encoder_output_channels, model.encoder_output_length)
        reconstructed = model.decoder(decoded_reshaped)
        
    # Visualize original vs reconstructed
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(single_example[0, 0].cpu().numpy())
    plt.title('Original Time Series')
    
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed[0, 0].cpu().numpy())
    plt.title('Reconstructed Time Series')
    
    plt.tight_layout()
    plt.savefig('Reconstruction.png')
    
    # 5. Bonus: Compare activations across different types of time series
    # If you have different classes or types of time series data
    if X.shape[0] > 50:  # If you have enough samples
        # Randomly select samples from different parts of your dataset
        indices1 = np.random.choice(X.shape[0]//3, 5)
        indices2 = np.random.choice(range(X.shape[0]//3, 2*X.shape[0]//3), 5)
        indices3 = np.random.choice(range(2*X.shape[0]//3, X.shape[0]), 5)
        
        group1 = X[indices1].to(DEVICE)
        group2 = X[indices2].to(DEVICE)
        group3 = X[indices3].to(DEVICE)
        
        # Compare activations on most important filter
        top_filter = top_filter_info['top_filters'][0]
        
        # Function to get activations for a batch
        def get_filter_activations(model, data, filter_idx):
            activations = {}
            
            def hook_fn(module, input, output):
                activations['features'] = output.detach()
            
            # Register hook to first conv layer
            for name, module in model.named_modules():
                if name == "encoder.0":
                    hook = module.register_forward_hook(hook_fn)
            
            # Forward pass
            with torch.no_grad():
                model.encoder(data)
            
            # Remove hook
            hook.remove()
            
            # Get activations for the specified filter
            filter_acts = activations['features'][:, filter_idx, :].cpu().numpy()
            return filter_acts
        
        # Get activations for each group
        acts1 = get_filter_activations(model, group1, top_filter)
        acts2 = get_filter_activations(model, group2, top_filter)
        acts3 = get_filter_activations(model, group3, top_filter)
        
        # Plot comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        for i in range(acts1.shape[0]):
            plt.plot(acts1[i])
        plt.title(f'Filter {top_filter} Activations - Group 1')
        
        plt.subplot(3, 1, 2)
        for i in range(acts2.shape[0]):
            plt.plot(acts2[i])
        plt.title(f'Filter {top_filter} Activations - Group 2')
        
        plt.subplot(3, 1, 3)
        for i in range(acts3.shape[0]):
            plt.plot(acts3[i])
        plt.title(f'Filter {top_filter} Activations - Group 3')
        
        plt.tight_layout()
        plt.savefig('./filters/Activations.png')
