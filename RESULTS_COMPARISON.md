# RESULTS_COMPARISON.md — Conditioned vs Non-Conditioned Baseline

Both models: 30D latent, LSTM-VAE, 256 hidden, 2 layers, identical training
schedule. Only difference: `use_scale_conditioning`.

Headline: the architectural innovation is responsible for the max-value
prediction lift; everything else is comparable.

| Metric | Conditioned | Baseline (no cond.) | Δ |
|---|---|---|---|
| Recon $R^2$ (normalized) | 0.9335 | 0.9449 | -0.0114 ▼ |
| Recon $R^2$ (original scale) | 0.9654 | 0.8444 | +0.1211 ▲ |
| Recon MAE (normalized) | 0.0541 | 0.0490 | +0.0051 ▼ |
| Max-value $R^2$ (curves 1–6) | 0.9711 | 0.5905 | +0.3806 ▲ |

| Per-curve recon $R^2$ | x0 | x1 | x2 | x3 | x4 | x5 | x6 |
|---|---|---|---|---|---|---|---|
| Conditioned | 0.936 | 0.935 | 0.933 | 0.918 | 0.931 | 0.938 | 0.936 |
| Baseline    | 0.946 | 0.946 | 0.943 | 0.943 | 0.942 | 0.943 | 0.946 |

| Per-curve max-val $R^2$ (curves 1–6) | x1 | x2 | x3 | x4 | x5 | x6 |
|---|---|---|---|---|---|---|
| Conditioned | 0.946 | 0.946 | 0.928 | 0.907 | 0.869 | 0.783 |
| Baseline    | -0.308 | -0.289 | -0.226 | -0.146 | -0.043 | 0.036 |

## Latent space

| Metric | Conditioned | Baseline |
|---|---|---|
| Active dims (var ≥ 0.01) | 25 / 30 | 30 / 30 |
| Collapsed dims | 5 / 30 | 0 / 30 |
| PCs for 90% / 95% / 99% var | 20 / 22 / 23 | 25 / 27 / 30 |
