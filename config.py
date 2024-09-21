hp = {
    "lr": 0.001, 
    "epochs": 200,
    "batch_size": 100, # Set to -1 to use the full dataset (maximum) batch size. 
    "weight_decay":0
}


model_config = {
    "in_channels" : 1, 
    "channels": 30, 
    "depth": 5, # Passed as a CLI argument above
    "reduced_size" : 30,
    "out_channels" : 10,  # Vary the embedding dimension of the dataset, normally pased as a CLI argument. 
    "kernel_size": 3,
    "window_length": 134,
}