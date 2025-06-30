import torch
import torch.nn as nn

# You can keep the activation consistent with your other models
ACTIVATION = nn.SiLU()

class ConditionalRecurrentVAE(nn.Module):
    """
    A Conditional Recurrent VAE using LSTMs to model and generate time series data
    based on a provided behavioral fingerprint.

    This version implements the robust architectural pattern of injecting the latent
    vector `z` at every timestep of the decoder to prevent information loss over
    long sequences.
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        
        # Core model parameters from config
        self.n_curves = config.get('in_channels', 7)
        self.seq_len = config.get('seq_len', 65)
        self.latent_dim = config['latent_dim']
        print(f'Latent dim: {self.latent_dim}')
        self.fingerprint_dim = config['fingerprint_dim'] # Should be 3
        
        # Hyperparameters for the LSTMs
        self.rnn_hidden_size = config.get('rnn_hidden_size', 256) # Kept your larger size
        self.rnn_num_layers = config.get('rnn_num_layers', 3)   # Kept your larger size
        print(f'{self.rnn_hidden_size=}, {self.rnn_num_layers=}')
        
        # --- 1. Encoder (No changes here) ---
        encoder_input_dim = self.n_curves + self.fingerprint_dim
        self.encoder_rnn = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
            bidirectional=True
        )
        encoder_output_dim = self.rnn_num_layers * 2 * self.rnn_hidden_size

        # --- 2. Bottleneck (No changes here) ---
        self.fc_mu = nn.Linear(encoder_output_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_dim, self.latent_dim)
        
        self.parameter_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            ACTIVATION,
            nn.Linear(64, self.fingerprint_dim)
        )

        # --- 3. Decoder ---
        # The decoder's initial state is still conditioned on z and y
        decoder_initial_state_dim = self.rnn_num_layers * self.rnn_hidden_size
        self.latent_to_hidden = nn.Linear(self.latent_dim + self.fingerprint_dim, decoder_initial_state_dim)
        self.latent_to_cell = nn.Linear(self.latent_dim + self.fingerprint_dim, decoder_initial_state_dim)

        self.encoder_dropout = nn.Dropout(p=0.4)

        # --- ARCHITECTURAL FIX 1: Update Decoder Input Dimension ---
        # The decoder LSTM input now also includes `z` at every step.
        decoder_input_dim = self.n_curves + self.fingerprint_dim + self.latent_dim
        self.decoder_rnn = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True
        )
        
        self.output_map = nn.Linear(self.rnn_hidden_size, self.n_curves)

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X: torch.Tensor, y: torch.Tensor, teacher_forcing_ratio=0.5) -> tuple:
        batch_size = X.size(0)
        X_rnn = X.permute(0, 2, 1)
        y_step_input = y.unsqueeze(1)
        
        # --- Encoding ---
        y_expanded_for_encoder = y.unsqueeze(1).expand(-1, self.seq_len, -1)
        encoder_input = torch.cat([X_rnn, y_expanded_for_encoder], dim=-1)
        _, (h_n, _) = self.encoder_rnn(encoder_input)
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        # encoded_summary = self.encoder_dropout(encoded_summary)
        
        # --- Bottleneck ---
        mu = self.fc_mu(encoded_summary)
        log_var = self.fc_log_var(encoded_summary)
        z = self.sample(mu, log_var)
        y_hat = self.parameter_predictor(mu)
        
        # --- Decoding ---
        # 1. Initialize Decoder State
        z_and_y = torch.cat([z, y], dim=-1)
        h_0 = self.latent_to_hidden(z_and_y).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        c_0 = self.latent_to_cell(z_and_y).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        hidden_state = (h_0, c_0)

        # 2. Prepare inputs for the autoregressive loop
        current_curve_input = torch.zeros(batch_size, 1, self.n_curves, device=X.device)
        outputs = []
        
        # --- ARCHITECTURAL FIX 2: Prepare `z` for step-by-step injection ---
        # Reshape z to be concatenated at each time step: (N, Z) -> (N, 1, Z)
        z_step_input = z.unsqueeze(1)

        # 3. Autoregressive loop
        for t in range(self.seq_len):
            # --- ARCHITECTURAL FIX 3: Inject z into the input at every step ---
            # The input is now [last_output, condition_y, global_z]
            step_input = torch.cat([current_curve_input, y_step_input, z_step_input], dim=-1)
            
            output, hidden_state = self.decoder_rnn(step_input, hidden_state)
            output = self.output_map(output) # Shape: (N, 1, C)
            outputs.append(output)
            
            use_teacher_force = torch.rand(1) < teacher_forcing_ratio
            if self.training and use_teacher_force:
                current_curve_input = X_rnn[:, t, :].unsqueeze(1)
            else:
                current_curve_input = output

        X_hat_rnn = torch.cat(outputs, dim=1)
        X_hat = X_hat_rnn.permute(0, 2, 1)
        X_hat = torch.clamp(X_hat, min=0)

        return X_hat, mu, log_var, y_hat

    def generate(self, fingerprint: torch.Tensor, num_samples: int, device: torch.device) -> torch.Tensor:
        self.eval() 

        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            y = fingerprint.unsqueeze(0).expand(num_samples, -1)

            # --- Initialize Decoder State ---
            z_and_y = torch.cat([z, y], dim=-1)
            h_0 = self.latent_to_hidden(z_and_y).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
            c_0 = self.latent_to_cell(z_and_y).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
            hidden_state = (h_0, c_0)
            
            # --- Autoregressive Loop ---
            current_curve_input = torch.zeros(num_samples, 1, self.n_curves).to(device)
            y_step_input = y.unsqueeze(1)

            # --- ARCHITECTURAL FIX 4: Prepare z for step-by-step injection in generation ---
            z_step_input = z.unsqueeze(1)
            
            generated_steps = []
            for _ in range(self.seq_len):
                # --- ARCHITECTURAL FIX 5: Inject z here as well ---
                step_input = torch.cat([current_curve_input, y_step_input, z_step_input], dim=-1)
                
                output, hidden_state = self.decoder_rnn(step_input, hidden_state)
                output = self.output_map(output)
                
                current_curve_input = output
                generated_steps.append(output)

            generated_sequences = torch.cat(generated_steps, dim=1)
            generated_sequences = generated_sequences.permute(0, 2, 1)
            return torch.clamp(generated_sequences, min=0)
