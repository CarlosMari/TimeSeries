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
        # The number of max values we want to predict
        # NOTE: Curve 0's max is always 1.0 (normalization artifact), so we only predict curves 1-6
        self.max_value_dim = self.n_curves - 1  # Predict 6 values instead of 7

        # Scale prediction mode: 'linear', 'log', or 'exp'
        self.scale_prediction_mode = config.get('scale_prediction_mode', 'linear')

        # NEW: Scale conditioning flag
        self.use_scale_conditioning = config.get('use_scale_conditioning', False)

        print(f"VAE Parameters: Latent Dim={self.latent_dim}, Num Curves={self.n_curves}")
        print(f"Scale Prediction Mode: {self.scale_prediction_mode}")
        print(f"Scale Conditioning: {'ENABLED' if self.use_scale_conditioning else 'DISABLED'}")

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

        # --- NEW: Scale Conditioning Components ---
        if self.use_scale_conditioning:
            # Project BiLSTM output to z_shape (latent_dim)
            self.shape_projector = nn.Linear(encoder_output_dim, self.latent_dim)

            # Encode max_vals to z_scale (latent_dim)
            self.scale_encoder = nn.Sequential(
                nn.Linear(self.n_curves, self.latent_dim),
                ACTIVATION,
                nn.Linear(self.latent_dim, self.latent_dim)
            )

            # Bottleneck takes concatenated [z_shape, z_scale]
            bottleneck_input_dim = self.latent_dim * 2
            print(f"  Conditioning: z_shape ({self.latent_dim}D) + z_scale ({self.latent_dim}D) → z ({self.latent_dim}D)")
        else:
            # Original: bottleneck takes encoder output directly
            bottleneck_input_dim = encoder_output_dim

        # --- 2. Bottleneck (Reparameterization) ---
        self.fc_mu = nn.Linear(bottleneck_input_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(bottleneck_input_dim, self.latent_dim)

        # --- NEW: MLP to predict max values from the latent space ---
        # Note: Output is in transformed space (log or exp) depending on scale_prediction_mode
        # When conditioning is enabled, this operates on z_shape (before scale conditioning)
        # When disabled, this operates on z (after sampling)
        self.max_value_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim//2),
            nn.Dropout(0.2),
            ACTIVATION,
            nn.Linear(self.latent_dim//2, self.max_value_dim)
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

    def transform_max_values(self, max_vals):
        """Transform max values from original space to prediction space."""
        if self.scale_prediction_mode == 'log':
            # Add small epsilon to avoid log(0)
            return torch.log(max_vals + 1e-8)
        elif self.scale_prediction_mode == 'exp':
            return torch.exp(max_vals)
        else:  # 'linear'
            return max_vals

    def inverse_transform_max_values(self, transformed_vals):
        """Transform predicted values back to original max value space."""
        if self.scale_prediction_mode == 'log':
            return torch.exp(transformed_vals)
        elif self.scale_prediction_mode == 'exp':
            return torch.log(torch.clamp(transformed_vals, min=1e-8))
        else:  # 'linear'
            return transformed_vals

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X: torch.Tensor, max_vals: torch.Tensor = None, teacher_forcing_ratio=0.5) -> tuple:
        """
        Args:
            X (torch.Tensor): The input data batch. Shape: (N, C, L) -> (N, 7, 65)
            max_vals (torch.Tensor, optional): Ground truth max values for conditioning. Shape: (N, 7)
                Required when use_scale_conditioning=True. Defaults to None.
            teacher_forcing_ratio (float): The probability of using teacher forcing.

        Returns:
            tuple: (X_hat, mu, log_var, max_vals_pred, max_vals_pred_transformed)
                - X_hat (torch.Tensor): The reconstructed time series.
                - mu (torch.Tensor): The latent mean.
                - log_var (torch.Tensor): The latent log variance.
                - max_vals_pred (torch.Tensor): The predicted max values (original scale).
                - max_vals_pred_transformed (torch.Tensor): Predicted max values (transformed space).
        """
        batch_size = X.size(0)
        # Permute X to (N, L, C) for LSTM compatibility
        X_rnn = X.permute(0, 2, 1)

        # --- Encoding ---
        # Encode the sequence X with BiLSTM
        _, (h_n, _) = self.encoder_rnn(X_rnn)
        # Concatenate final hidden states from both directions
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # --- Prepare bottleneck input ---
        if self.use_scale_conditioning:
            # Check that max_vals is provided
            if max_vals is None:
                raise ValueError("max_vals must be provided when use_scale_conditioning=True")

            # Project BiLSTM output to z_shape
            z_shape = self.shape_projector(encoded_summary)  # (N, latent_dim)

            # Encode max_vals to z_scale
            z_scale = self.scale_encoder(max_vals)  # (N, latent_dim)

            # Concatenate for VAE bottleneck
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)  # (N, 2*latent_dim)
        else:
            # No scale conditioning - original architecture
            bottleneck_input = encoded_summary

        # --- VAE Bottleneck ---
        mu = self.fc_mu(bottleneck_input)
        log_var = self.fc_log_var(bottleneck_input)
        z = self.sample(mu, log_var)

        # --- Parallel Prediction Head ---
        # ALWAYS predict from z (the latent space) - proper VAE structure!
        # During training: z ~ q(z|X, max_vals) - contains scale information
        # During generation: z ~ p(z) = N(0,I) - sampled from prior
        max_vals_pred_transformed_partial = self.max_value_predictor(z)

        # Convert back to original scale
        max_vals_pred_partial = self.inverse_transform_max_values(max_vals_pred_transformed_partial)  # Shape: (N, 6)

        # Prepend curve 0's max value (always 1.0)
        ones = torch.ones(batch_size, 1, device=z.device)
        max_vals_pred = torch.cat([ones, max_vals_pred_partial], dim=1)  # Shape: (N, 7)

        # For transformed space (used in loss computation)
        ones_transformed = self.transform_max_values(ones)
        max_vals_pred_transformed = torch.cat([ones_transformed, max_vals_pred_transformed_partial], dim=1)
        
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

        return X_hat, mu, log_var, max_vals_pred, max_vals_pred_transformed

    def generate(self, num_samples: int, device: torch.device) -> tuple:
        """
        Generates new time series samples unconditionally.

        Proper VAE generation:
          1. Sample z from prior: z ~ N(0,I)
          2. Predict max_vals from z
          3. Decode X from z

        During training, z learned to encode both shape AND scale (via conditioning).
        During generation, we simply sample from the prior and predict everything from z.

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
            # Sample latent vectors from standard normal prior
            z = torch.randn(num_samples, self.latent_dim, device=device)

            # Predict max values from z (curves 1-6; curve 0 is always 1.0)
            predicted_max_vals_transformed_partial = self.max_value_predictor(z)  # Shape: (N, 6)
            predicted_max_vals_partial = self.inverse_transform_max_values(predicted_max_vals_transformed_partial)

            # Prepend curve 0's max value (always 1.0)
            ones = torch.ones(num_samples, 1, device=device)
            predicted_max_vals = torch.cat([ones, predicted_max_vals_partial], dim=1)  # Shape: (N, 7)

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
        

    def decode(self, z: torch.Tensor) -> tuple:
        """
        Decodes a specific latent vector `z` into a time series.

        Proper VAE decoding: Everything is predicted from z (the latent space).

        Args:
            z (torch.Tensor): The latent vector(s) to decode. Shape: (N, latent_dim).

        Returns:
            tuple: A tuple containing:
                - generated_sequences (torch.Tensor): The new time series.
                - predicted_max_vals (torch.Tensor): The predicted max values.
        """
        self.eval()
        num_samples = z.size(0)
        device = z.device

        with torch.no_grad():
            # Predict max values for curves 1-6 from z (curve 0 is always 1.0)
            predicted_max_vals_transformed_partial = self.max_value_predictor(z)  # Shape: (N, 6)
            predicted_max_vals_partial = self.inverse_transform_max_values(predicted_max_vals_transformed_partial)

            # Prepend curve 0's max value (always 1.0)
            ones = torch.ones(num_samples, 1, device=device)
            predicted_max_vals = torch.cat([ones, predicted_max_vals_partial], dim=1)  # Shape: (N, 7)

            # --- Initialize Decoder State from z ---
            h_0 = self.latent_to_hidden(z).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
            c_0 = self.latent_to_cell(z).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
            hidden_state = (h_0, c_0)
            
            # --- Autoregressive Loop ---
            current_curve_input = torch.zeros(num_samples, 1, self.n_curves).to(device)
            z_step_input = z.unsqueeze(1)
            
            generated_steps = []
            for _ in range(self.seq_len):
                step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
                output, hidden_state = self.decoder_rnn(step_input, hidden_state)
                output = self.output_map(output)
                current_curve_input = output
                generated_steps.append(output)

            generated_sequences_rnn = torch.cat(generated_steps, dim=1)
            generated_sequences = generated_sequences_rnn.permute(0, 2, 1)
            generated_sequences = torch.clamp(generated_sequences, min=0, max=1)

            return generated_sequences, predicted_max_vals