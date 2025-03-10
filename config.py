hp = {
    "lr": 0.005, 
    "epochs": 15000,
    "batch_size": 1000, # Set to -1 to use the full dataset (maximum) batch size. 
    "weight_decay":0,
    "random_seed": 73,
    "alpha": 1e-2,
}


model_config = {
    "in_channels" : 7, 
    "channels": 30, 
    "depth": 5, # Passed as a CLI argument above
    "reduced_size" : 30,
    "latent_dim" : 10,  # Vary the embedding dimension of the dataset, normally pased as a CLI argument. 
    "kernel_size": 3,
    "input_size": 134,
    "save": True,
    "save_route": './model_ckpts/',
    "name": "model_alpha_weight_anneal",
    "sampling": 1, # Sample the input, maximum 1.
}


DEVICE = 'cuda'