hp = {
    "lr": 1e-4, 
    "epochs": 2000,
    "batch_size": 1000, # Set to -1 to use the full dataset (maximum) batch size. 
    "weight_decay":0,
    "random_seed": 73,
    "alpha": 1e-2,
    'gamma': 0,         # Weight for the fingerprint loss
    'warmup_epochs': 300,
    'beta_max': 2e-4,
    'lambda_max_val': 0.5, 
}


model_config = {
    "in_channels" : 7,
    "channels": 30,
    "depth": 5, # Passed as a CLI argument above
    "reduced_size" : 30,
    "latent_dim" : 50,  # Vary the embedding dimension of the dataset, normally pased as a CLI argument.
    "kernel_size": 3,
    "input_size": 134,
    "save": True,
    "save_route": './model_ckpts/',
    "name": "model_final",
    "sampling": 1, # Sample the input, maximum 1.
    'rnn_hidden_size': 256,
    'rnn_num_layers': 2,
    'scale_prediction_mode': 'log',  # Options: 'linear', 'log', 'exp' - Controls how max values are predicted
}


DEVICE = 'cuda'
