import torch
import torch.nn as nn

# You can keep the activation consistent with your other models
ACTIVATION = nn.SiLU()

class LSTM_VAE(nn.Module):
    """
    An LSTM-based Variational Autoencoder for generating time series data.

    This model encodes a sequence into a latent distribution (mu, log_var),
    samples a latent vector `z`, and then decodes `z` back into a sequence.
    
    A key feature is a parallel MLP head that predicts the maximum value of each 
    curve in the sequence directly from the latent vector `z`. This helps the
    latent space learn structurally important features of the time series.
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        
        # --- Core model parameters from config ---
        self.n_curves = config.get('n_curves', 7)
        self.seq_len = config.get('seq_len', 65)
        self.latent_dim = config['latent_dim']
        # The number of max values we want to predict (one for each curve)
        self.max_value_dim = self.n_curves
        
        print(f"VAE Parameters: Latent Dim={self.latent_dim}, Num Curves={self.n_curves}")
        
        # Hyperparameters for the LSTMs
        self.rnn_hidden_size = config.get('rnn_hidden_size', 256)
        self.rnn_num_layers = config.get('rnn_num_layers', 3)
        print(f"LSTM Parameters: Hidden Size={self.rnn_hidden_size}, Num Layers={self.rnn_num_layers}")
        
        # --- 1. Encoder ---
        # The encoder input is now just the time series data itself.
        self.encoder_rnn = nn.LSTM(
            input_size=self.n_curves,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
            bidirectional=True
        )
        # The output dimension from the concatenated final hidden states of the bidirectional LSTM
        encoder_output_dim = self.rnn_num_layers * 2 * self.rnn_hidden_size

        # --- 2. Bottleneck (Reparameterization) ---
        self.fc_mu = nn.Linear(encoder_output_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_dim, self.latent_dim)
        
        # --- NEW: MLP to predict max values from the latent space ---
        self.max_value_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 30),
            nn.Dropout(0.2),
            ACTIVATION,
            nn.Linear(30, 30),
            nn.Dropout(0.2),
            ACTIVATION,
            nn.Linear(30, self.max_value_dim)
        )

        # --- 3. Decoder ---
        # The decoder's initial state is conditioned only on the latent vector `z`.
        decoder_initial_state_dim = self.rnn_num_layers * self.rnn_hidden_size
        self.latent_to_hidden = nn.Linear(self.latent_dim, decoder_initial_state_dim)
        self.latent_to_cell = nn.Linear(self.latent_dim, decoder_initial_state_dim)

        self.encoder_dropout = nn.Dropout(p=0.4)

        # The decoder LSTM input now includes `z` at every step to guide generation.
        decoder_input_dim = self.n_curves + self.latent_dim
        self.decoder_rnn = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True
        )
        
        # Final layer to map decoder's hidden state to the curve values
        self.output_map = nn.Linear(self.rnn_hidden_size, self.n_curves)

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X: torch.Tensor, teacher_forcing_ratio=0.5) -> tuple:
        """
        Args:
            X (torch.Tensor): The input data batch. Shape: (N, C, L) -> (N, 7, 65)
            teacher_forcing_ratio (float): The probability of using teacher forcing.

        Returns:
            tuple: A tuple containing:
                - X_hat (torch.Tensor): The reconstructed time series.
                - mu (torch.Tensor): The latent mean.
                - log_var (torch.Tensor): The latent log variance.
                - max_vals_pred (torch.Tensor): The predicted max values for each curve.
        """
        batch_size = X.size(0)
        # Permute X to (N, L, C) for LSTM compatibility
        X_rnn = X.permute(0, 2, 1)
        
        # --- Encoding ---
        # No more 'y' condition, just encode the sequence X
        _, (h_n, _) = self.encoder_rnn(X_rnn)
        # Concatenate final hidden states from both directions
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        # encoded_summary = self.encoder_dropout(encoded_summary)
        
        # --- Bottleneck ---
        mu = self.fc_mu(encoded_summary)
        log_var = self.fc_log_var(encoded_summary)
        z = self.sample(mu, log_var)
        
        # --- Parallel Prediction Head ---
        # Predict max values from the sampled latent vector z
        max_vals_pred = self.max_value_predictor(z)
        
        # --- Decoding ---
        # 1. Initialize Decoder State from latent vector z
        h_0 = self.latent_to_hidden(z).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        c_0 = self.latent_to_cell(z).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        hidden_state = (h_0, c_0)

        # 2. Prepare inputs for the autoregressive loop
        current_curve_input = torch.zeros(batch_size, 1, self.n_curves, device=X.device)
        outputs = []
        
        # Reshape z for step-by-step injection: (N, Z) -> (N, 1, Z)
        z_step_input = z.unsqueeze(1)

        # 3. Autoregressive loop
        for t in range(self.seq_len):
            # Input is now [last_output, global_z]
            step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
            
            output, hidden_state = self.decoder_rnn(step_input, hidden_state)
            output = self.output_map(output) # Shape: (N, 1, C)
            outputs.append(output)
            
            use_teacher_force = torch.rand(1) < teacher_forcing_ratio
            if self.training and use_teacher_force:
                current_curve_input = X_rnn[:, t, :].unsqueeze(1)
            else:
                current_curve_input = output

        X_hat_rnn = torch.cat(outputs, dim=1)
        # Permute back to (N, C, L)
        X_hat = X_hat_rnn.permute(0, 2, 1)
        X_hat = torch.clamp(X_hat, min=0, max=1) # Clamp to [0,1] since input is normalized

        return X_hat, mu, log_var, max_vals_pred

    def generate(self, num_samples: int, device: torch.device) -> tuple:
        """
        Generates new time series samples unconditionally.

        Args:
            num_samples (int): The number of samples to generate.
            device (torch.device): The device to perform generation on (e.g., 'cuda').

        Returns:
            tuple: A tuple containing:
                - generated_sequences (torch.Tensor): The new time series.
                - predicted_max_vals (torch.Tensor): The predicted max values for the generated curves.
        """
        self.eval() 

        with torch.no_grad():
            # Sample latent vectors from a standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # Predict the corresponding max values for these new samples
            predicted_max_vals = self.max_value_predictor(z)

            # --- Initialize Decoder State from z ---
            h_0 = self.latent_to_hidden(z).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
            c_0 = self.latent_to_cell(z).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
            hidden_state = (h_0, c_0)
            
            # --- Autoregressive Loop ---
            current_curve_input = torch.zeros(num_samples, 1, self.n_curves).to(device)
            # Reshape z for step-by-step injection
            z_step_input = z.unsqueeze(1)
            
            generated_steps = []
            for _ in range(self.seq_len):
                # Input is [last_output, global_z]
                step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
                
                output, hidden_state = self.decoder_rnn(step_input, hidden_state)
                output = self.output_map(output)
                
                current_curve_input = output
                generated_steps.append(output)

            generated_sequences_rnn = torch.cat(generated_steps, dim=1)
            # Permute back to (N, C, L)
            generated_sequences = generated_sequences_rnn.permute(0, 2, 1)
            generated_sequences = torch.clamp(generated_sequences, min=0, max=1)

            return generated_sequences, predicted_max_vals