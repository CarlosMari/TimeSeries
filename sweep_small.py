"""
Small-scale hyperparameter sweep for LSTM_VAE.

Tests different configurations:
- latent_dim: [20, 30, 40, 50]
- scale_prediction_mode: ['linear', 'log', 'exp']

Each configuration runs for 200 epochs to quickly identify best settings.
"""

import torch
import wandb
import numpy as np
from itertools import product

# Import your training setup
from train_cvae import train, TimeSeriesDataset
from VAE.models.cvae import LSTM_VAE
from config import hp, model_config, DEVICE


def run_sweep_manual():
    """
    Manual sweep testing latent dimensions and scale prediction modes.
    """

    # Define configurations to test
    latent_dims = [20, 30, 40, 50]
    scale_modes = ['linear', 'log']  # Removed 'exp' - doesn't work

    configurations = list(product(latent_dims, scale_modes))

    # Find the index to start from (linear, 40)
    start_config = ('linear', 40)  # Note: order is (scale_mode, latent_dim) in tuple
    # But product gives (latent_dim, scale_mode), so we need to find (40, 'linear')
    start_idx = 0
    for i, (ld, sm) in enumerate(configurations):
        if ld == 40 and sm == 'linear':
            start_idx = i
            break

    # Only run from start_idx onwards
    configurations = configurations[start_idx:]

    print(f"\n{'='*60}")
    print(f"Running {len(configurations)} configurations (200 epochs each)")
    print(f"Starting from: latent_dim=40, scale_mode='linear'")
    print(f"Remaining configs: {configurations}")
    print(f"{'='*60}\n")

    results = []

    for config_idx, (latent_dim, scale_mode) in enumerate(configurations, 1):
        print(f"\n{'='*60}")
        print(f"Configuration {config_idx}/{len(configurations)}")
        print(f"latent_dim={latent_dim}, scale_mode={scale_mode}")
        print(f"{'='*60}\n")

        # Initialize wandb for this run
        run_name = f"sweep_ld{latent_dim}_scale{scale_mode}"
        wandb.init(
            project='Conditional_LV_VAE_Sweep',
            name=run_name,
            config={
                'latent_dim': latent_dim,
                'scale_prediction_mode': scale_mode,
                'epochs': 200,
            },
            reinit=True
        )

        # Update model config
        current_model_config = model_config.copy()
        current_model_config['latent_dim'] = latent_dim
        current_model_config['scale_prediction_mode'] = scale_mode
        current_model_config['name'] = run_name
        current_model_config['save'] = True  # Save all sweep models

        # Update hyperparameters
        current_hp = hp.copy()
        current_hp['epochs'] = 200

        # Log full config to wandb
        wandb.config.update({
            'rnn_hidden_size': current_model_config['rnn_hidden_size'],
            'rnn_num_layers': current_model_config['rnn_num_layers'],
            'lambda_max_val': current_hp['lambda_max_val'],
            'beta_max': current_hp['beta_max'],
            'lr': current_hp['lr'],
            'batch_size': current_hp['batch_size'],
        })

        # Create and train model
        model = LSTM_VAE(config=current_model_config)

        # Temporarily modify globals for train() function
        import config
        original_model_config = config.model_config
        original_hp = config.hp

        config.model_config = current_model_config
        config.hp = current_hp

        try:
            trained_model = train(model)

            # Get final metrics from wandb
            api = wandb.Api()
            run_data = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")

            final_recon_loss = run_data.summary.get('Test Recon Loss', None)
            final_max_val_loss = run_data.summary.get('Test Max Val Loss', None)
            final_coverage = run_data.summary.get('Test Overall Coverage', None)

            results.append({
                'latent_dim': latent_dim,
                'scale_mode': scale_mode,
                'final_recon_loss': final_recon_loss,
                'final_max_val_loss': final_max_val_loss,
                'final_coverage': final_coverage,
                'run_name': run_name,
            })

            print(f"\n✓ Completed: recon_loss={final_recon_loss:.6f}, max_val_loss={final_max_val_loss:.6f}")

        except Exception as e:
            print(f"\n✗ ERROR in configuration: {e}")
            results.append({
                'latent_dim': latent_dim,
                'scale_mode': scale_mode,
                'final_recon_loss': None,
                'final_max_val_loss': None,
                'final_coverage': None,
                'error': str(e),
            })

        finally:
            # Restore original config
            config.model_config = original_model_config
            config.hp = original_hp
            wandb.finish()

    # Print summary
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    print(f"{'latent_dim':<12} {'scale_mode':<12} {'Recon Loss':<15} {'MaxVal Loss':<15} {'Coverage':<10}")
    print("-"*80)

    # Sort by reconstruction loss (best first)
    valid_results = [r for r in results if r['final_recon_loss'] is not None]
    valid_results.sort(key=lambda x: x['final_recon_loss'])

    for r in valid_results:
        recon = r['final_recon_loss']
        maxval = r['final_max_val_loss'] if r['final_max_val_loss'] else 0
        cov = r['final_coverage'] if r['final_coverage'] else 0
        print(f"{r['latent_dim']:<12} {r['scale_mode']:<12} {recon:<15.6f} {maxval:<15.6f} {cov:<10.4f}")

    if valid_results:
        print("\n" + "="*80)
        print("TOP 3 CONFIGURATIONS (by reconstruction loss):")
        print("="*80)
        for i, r in enumerate(valid_results[:3], 1):
            print(f"\n{i}. latent_dim={r['latent_dim']}, scale_mode={r['scale_mode']}")
            print(f"   Recon Loss: {r['final_recon_loss']:.6f}")
            print(f"   MaxVal Loss: {r['final_max_val_loss']:.6f}")
            print(f"   Coverage: {r['final_coverage']:.4f}")

        best = valid_results[0]
        print("\n" + "="*80)
        print("BEST OVERALL CONFIGURATION:")
        print(f"  latent_dim = {best['latent_dim']}")
        print(f"  scale_prediction_mode = '{best['scale_mode']}'")
        print(f"  Test Recon Loss = {best['final_recon_loss']:.6f}")
        print(f"  Test MaxVal Loss = {best['final_max_val_loss']:.6f}")
        print(f"  Coverage = {best['final_coverage']:.4f}")
        print("="*80)

    # Save results to file
    import json
    with open('sweep_results_200ep.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to sweep_results_200ep.json")

    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Small-scale Hyperparameter Sweep (RESUMED)")
    print("="*60)
    print("\nTesting:")
    print("  - latent_dim: [20, 30, 40, 50]")
    print("  - scale_prediction_mode: ['linear', 'log']  (exp removed)")
    print("  - 200 epochs per configuration")
    print("  - Starting from: latent_dim=40, scale_mode='linear'")
    print("  - Remaining: 4 configurations")
    print("\nThis will take approximately:")
    print("  ~30-40 minutes (depending on GPU speed)")
    print("="*60 + "\n")

    input("Press Enter to start sweep...")

    results = run_sweep_manual()
