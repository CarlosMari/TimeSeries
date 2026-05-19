# REFERENCES.md — Citations for the paper

Living document. Each entry: bibtex-able key, full reference, **what we cite it for** in the paper. Sections track the paper structure under the 2026-05-15 pivot (`docs/superpowers/specs/2026-05-15-comparative-evaluation-design.md`).

---

## 1. Generalized Lotka–Volterra dynamics + ecology

- **Lotka 1925** — Lotka, A.J. *Elements of Physical Biology*. Williams & Wilkins, 1925. — original predator–prey formulation.
- **Volterra 1926** — Volterra, V. "Fluctuations in the abundance of a species considered mathematically." *Nature* 118: 558–560 (1926). — companion to Lotka, motivating the GLV form.
- **May 1972** — May, R.M. "Will a large complex system be stable?" *Nature* 238: 413–414 (1972). — eigenvalue stability of random ecological networks; the dynamical-stability lens underlying our (r, A) parameterization.
- **Allesina & Tang 2012** — Allesina, S., Tang, S. "Stability criteria for complex ecosystems." *Nature* 483: 205–208 (2012). — modern reference for random GLV stability; supports the data-generation rejection sampling.
- **Bunin 2017** — Bunin, G. "Ecological communities with Lotka-Volterra dynamics." *Phys. Rev. E* 95, 042414 (2017). — disordered-systems perspective; cite when discussing parameter-distribution choices (r ~ Exp(2), A_ii ~ −Exp(2), off-diag ~ N(0,1)).

## 2. Generative-model architectures we compare (models 1–7)

### LSTM-VAE family (models 1, 2, 3)
- **Kingma & Welling 2014** — Kingma, D.P., Welling, M. "Auto-Encoding Variational Bayes." *ICLR* 2014. arXiv:1312.6114. — VAE foundational.
- **Sohn et al. 2015** — Sohn, K., Lee, H., Yan, X. "Learning Structured Output Representation using Deep Conditional Generative Models." *NeurIPS* 2015. — CVAE, our scale-conditioning is in this family.
- **Bayer & Osendorfer 2014** — Bayer, J., Osendorfer, C. "Learning Stochastic Recurrent Networks." arXiv:1411.7610 (2014). — recurrent latent-variable models; predecessor for sequential VAEs.
- **Chung et al. 2015** — Chung, J., Kastner, K., Dinh, L., Goyal, A., Courville, A.C., Bengio, Y. "A Recurrent Latent Variable Model for Sequential Data." *NeurIPS* 2015. — VRNN; the conceptual template for our autoregressive decoder.
- **Higgins et al. 2017** — Higgins, I., et al. "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." *ICLR* 2017. — β annealing; we use β-warmup over 300 epochs.
- **Bowman et al. 2016** — Bowman, S.R., et al. "Generating Sentences from a Continuous Space." *CoNLL* 2016. — teacher-forcing schedule and KL-warmup tricks for sequence VAEs.

### Stochastic decoder (model 3 + B1 frozen-σ variants)
- **Goyal et al. 2017** — Goyal, A., Sordoni, A., Côté, M.-A., Ke, N.R., Bengio, Y. "Z-Forcing: Training Stochastic Recurrent Networks." *NeurIPS* 2017. — stochastic hidden state in recurrent generative models; principled basis for model 3.
- **Fraccaro et al. 2016** — Fraccaro, M., Sønderby, S.K., Paquet, U., Winther, O. "Sequential Neural Models with Stochastic Layers." *NeurIPS* 2016. — alternative stochastic-decoder formulation; methods comparison.
- *(Note: the B1 finding — that MSE recon collapses learned σ to zero, motivating the frozen-σ ablation — is, to our knowledge, novel; if a prior paper documents the same σ-collapse failure mode in seq2seq VAEs, we'll add it during the literature search before submission.)*

### VAE-implicit-denoising in time-series generation (root cause per §1.3.0 audit consolidation 2026-05-19)
- **Vincent et al. 2008** — Vincent, P., Larochelle, H., Bengio, Y., Manzagol, P.-A. "Extracting and Composing Robust Features with Denoising Autoencoders." *ICML* 2008. — foundational denoising-autoencoder paper; establishes that AE-style models trained on noisy data learn smoothed underlying signal as a feature.
- **Kingma & Welling 2014** — *(already cited under VAE foundations)*. The Gaussian-decoder VAE with MSE recon loss is a special case of a denoising objective: the encoder learns conditional mean of the clean signal given a noisy realization. This is the mechanism behind our §1.3.0 finding.
- **Salmona et al. 2022** — Salmona, A., De Bortoli, V., Delon, J., Desolneux, A. "Can Push-forward Generative Models Fit Multimodal Distributions?" *NeurIPS* 2022. — discusses how mode-covering / denoising behavior emerges in VAE-style generators; aligns with the §1.3.3.4.1 secondary-effect (dynamics-collapse) observation.
- **Lucas et al. 2019** — Lucas, J., Tucker, G., Grosse, R., Norouzi, M. "Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse." *NeurIPS* 2019. — links MSE recon + bottleneck capacity to posterior collapse / mode-collapse phenomena; supports the §1.3.3.4 mechanism.
- **(Future work direction)** Heteroscedastic-noise VAEs with explicit per-timestep predicted σ — the principled fix sketched in §1.3.0 action item 4. Candidate canonical reference: **Skafte et al. 2019** "Reliable training and estimation of variance networks" (NeurIPS) or any β-VAE/NLL variant that predicts variance alongside mean. To be filled in during literature search phase.

### Exposure bias / autoregressive train-test mismatch (secondary effect, see §1.3.3.3)
*Note: the audit found this to be a real but secondary effect (~30% of the gap), not the primary root cause. Retained as supporting literature for the LSTM-family-specific behavior we observe.*
- **Williams & Zipser 1989** — Williams, R.J., Zipser, D. "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks." *Neural Computation* 1(2):270–280 (1989). — original definition of teacher forcing; useful for paper framing of TF=1 vs TF=0.
- **Bengio et al. 2015** — Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N. "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks." *NeurIPS* 2015. arXiv:1506.03099. — the canonical exposure-bias / autoregressive-drift paper; introduces gradual TF decay schedules.
- **Ranzato et al. 2016** — Ranzato, M., Chopra, S., Auli, M., Zaremba, W. "Sequence Level Training with Recurrent Neural Networks." *ICLR* 2016. arXiv:1511.06732. — RL-based fix for exposure bias.
- **Goyal et al. 2017** — *(already in our §"Stochastic decoder" subsection above)* "Z-Forcing" — most directly relevant prior work on injecting stochasticity to break autoregressive drift; closest cousin of our σ=0.05 cure.

### Spectral / FFT-domain losses (B1 spectral-loss variant)
- **Engel et al. 2020** — Engel, J., Hantrakul, L., Gu, C., Roberts, A. "DDSP: Differentiable Digital Signal Processing." *ICLR* 2020. arXiv:2001.04643. — popularized multi-resolution STFT-magnitude loss for audio synthesis; our full-FFT magnitude-MSE (T=65, no windowing) is the simplest case of this family.
- **Yamamoto et al. 2020** — Yamamoto, R., Song, E., Kim, J.-M. "Parallel WaveGAN: A fast waveform generation model based on adversarial networks with multi-resolution spectrogram." *ICASSP* 2020. — multi-resolution spectrogram MSE; precedent for adding spectral terms to non-adversarial losses too.
- **Steinmetz & Reiss 2020** — Steinmetz, C.J., Reiss, J.D. "auraloss: Audio focused loss functions in PyTorch." *DMRN+15 Workshop* (2020). — implementation reference for spectral losses; we use a custom rfft-magnitude MSE in `train_cvae._spectral_loss()`.

### Latent-ODE (model 4)
- **Chen et al. 2018** — Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. "Neural Ordinary Differential Equations." *NeurIPS* 2018. — neural ODE foundational.
- **Rubanova et al. 2019** — Rubanova, Y., Chen, R.T.Q., Duvenaud, D. "Latent ODEs for Irregularly-Sampled Time Series." *NeurIPS* 2019. arXiv:1907.03907. — **the model-4 architecture spec.** ODE-RNN encoder + ODE prior + decoder.
- **Kidger et al. 2020** — Kidger, P., Morrill, J., Foster, J., Lyons, T. "Neural Controlled Differential Equations for Irregular Time Series." *NeurIPS* 2020. — comparator in literature for latent-ODE; cite to position our choice.

### Transformer-VAE (model 5)
- **Vaswani et al. 2017** — Vaswani, A., et al. "Attention Is All You Need." *NeurIPS* 2017. — Transformer foundational.
- **Wang et al. 2019** — Wang, T., Wan, X. "T-CVAE: Transformer-based Conditioned Variational Autoencoder for Story Completion." *IJCAI* 2019. — Transformer-VAE-style architecture.
- **Yang et al. 2020** — Yang, X., Wang, T., et al. "Transformer-based Conditional Variational Autoencoder for Controllable Story Generation." arXiv:2101.00828. — additional Transformer-VAE reference for the architectural choice.

### KAN-VAE (model 6)
- **Liu et al. 2024** — Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T.Y., Tegmark, M. "KAN: Kolmogorov–Arnold Networks." arXiv:2404.19756 (2024). — **the KAN paper; foundational for model 6.**
- **Efficient-KAN** — Blealtan / efficient-kan. https://github.com/Blealtan/efficient-kan. — implementation we use.
- **(TBC)** — any 2024–2025 paper using KAN as a VAE component, once we find one. If none, model 6 is one of the first uses of KAN in a sequential VAE — flag in paper.

### Direct GLV regression (model 7)
- **Stock et al. 2018** — Stock, A., et al. "Inferring ecological interactions from time-series data via the inverse Volterra problem." (we will cite the canonical inverse-GLV literature; specific reference TBC after lit search). — supports the inverse-problem baseline framing.
- **Maynard et al. 2020** — Maynard, D.S., Miller, Z.R., Allesina, S. "Predicting coexistence in experimental ecological communities." *Nature Ecology & Evolution* 4, 91–100 (2020). — practical inverse-problem on GLV; cite when motivating the baseline.

## 3. The 3-lens evaluation protocol

### Lens 1 — feature-space distributional fidelity (MMD + density/coverage)
- **Gretton et al. 2012** — Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., Smola, A. "A Kernel Two-Sample Test." *JMLR* 13:723–773 (2012). — **MMD with Gaussian kernel + permutation test; the foundational reference for Lens 1.**
- **Naeem et al. 2020** — Naeem, M.F., Oh, S.J., Uh, Y., Choi, Y., Yoo, J. "Reliable Fidelity and Diversity Metrics for Generative Models." *ICML* 2020. arXiv:2002.09797. — **density/coverage@k; we use exactly this formulation.**
- **Sajjadi et al. 2018** — Sajjadi, M.S.M., Bachem, O., Lucic, M., Bousquet, O., Gelly, S. "Assessing Generative Models via Precision and Recall." *NeurIPS* 2018. — precursor to Naeem; cite for context.

### Lens 2 — Recurrence Quantification Analysis
- **Eckmann et al. 1987** — Eckmann, J.-P., Kamphorst, S.O., Ruelle, D. "Recurrence Plots of Dynamical Systems." *Europhys. Lett.* 4(9):973 (1987). — **recurrence plots foundational.**
- **Zbilut & Webber 1992** — Zbilut, J.P., Webber, C.L. Jr. "Embeddings and delays as derived from quantification of recurrence plots." *Phys. Lett. A* 171(3-4):199–203 (1992). — RQA introduced.
- **Marwan et al. 2007** — Marwan, N., Romano, M.C., Thiel, M., Kurths, J. "Recurrence plots for the analysis of complex systems." *Physics Reports* 438(5-6):237–329 (2007). — **the canonical RQA review; cite for DET, L_max, LAM, TT definitions.**
- **Marwan 2011** — Marwan, N. "How to avoid potential pitfalls in recurrence plot based data analysis." *Int. J. Bifurcation and Chaos* 21(4):1003–1017 (2011). — supports our choice of fixed recurrence rate ε per trajectory.

### Lens 3 — Largest Lyapunov exponent
- **Rosenstein et al. 1993** — Rosenstein, M.T., Collins, J.J., De Luca, C.J. "A practical method for calculating largest Lyapunov exponents from small data sets." *Physica D* 65(1-2):117–134 (1993). — **the Rosenstein algorithm we implement.**
- **Wolf et al. 1985** — Wolf, A., Swift, J.B., Swinney, H.L., Vastano, J.A. "Determining Lyapunov exponents from a time series." *Physica D* 16(3):285–317 (1985). — alternative; cite as the historical predecessor we don't use due to short-sequence regime.
- **Kantz 1994** — Kantz, H. "A robust method to estimate the maximal Lyapunov exponent of a time series." *Phys. Lett. A* 185(1):77–87 (1994). — discussion of why for short noisy data Rosenstein is preferred.
- **Takens 1981** — Takens, F. "Detecting strange attractors in turbulence." In *Dynamical Systems and Turbulence* (Springer LNM 898). — **time-delay embedding (Takens' theorem); foundational for both RQA and Lyapunov on scalar time series.**

## 4. Generative-model evaluation for time series and dynamics

- **Heusel et al. 2017** — Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *NeurIPS* 2017. — FID; cite as the canonical generative-quality metric we are *not* using, with reason (Inception network is image-specific).
- **Esteban et al. 2017** — Esteban, C., Hyland, S.L., Rätsch, G. "Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs." arXiv:1706.02633. — RCGAN; commonly cited time-series generative-model paper; positioning.
- **Yoon et al. 2019** — Yoon, J., Jarrett, D., van der Schaar, M. "Time-series Generative Adversarial Networks." *NeurIPS* 2019. — TimeGAN; another standard time-series generative comparator. Their eval methodology (discriminative + predictive scores) is what we improve on.
- **Jeha et al. 2022** — Jeha, P., et al. "PSA-GAN: Progressive Self Attention GANs for Synthetic Time Series." *ICLR* 2022. — recent time-series generation eval landscape.

## 5. Preprocessing and normalization

- **Sun et al. 2024** — Sun, J., et al. "Family normalization for multi-species ecological time series." (TBC — placeholder; we may not need an external reference for our 3-stage pipeline since the design is custom.)
- **Hyndman & Athanasopoulos 2018** — Hyndman, R.J., Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.). OTexts. — chapter on transformations for time-series. Light reference; supports our log-space scale prediction head.

## 6. Statistical testing

- **Wilcoxon 1945** — Wilcoxon, F. "Individual Comparisons by Ranking Methods." *Biometrics Bulletin* 1(6):80–83 (1945). — Wilcoxon signed-rank for paired model comparisons.
- **Benjamini & Hochberg 1995** — Benjamini, Y., Hochberg, Y. "Controlling the False Discovery Rate." *J. R. Stat. Soc. B* 57(1):289–300 (1995). — FDR correction over the comparative metric grid.
- **Efron 1979** — Efron, B. "Bootstrap Methods: Another Look at the Jackknife." *Annals of Statistics* 7(1):1–26 (1979). — bootstrap CIs (we already use this in `RESULTS.json`).
- **Massey 1951** — Massey, F.J. "The Kolmogorov-Smirnov test for goodness of fit." *J. Am. Stat. Assoc.* 46(253):68–78 (1951). — KS test foundational.

## 7. Software / libraries

- **PyTorch** — Paszke, A., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS* 2019.
- **torchdiffeq** — Chen, R.T.Q. https://github.com/rtqichen/torchdiffeq. — for model 4.
- **SciPy** — Virtanen, P., et al. "SciPy 1.0: fundamental algorithms for scientific computing." *Nature Methods* 17:261–272 (2020). — `solve_ivp`, `stats.ks_2samp`.
- **scikit-learn** — Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." *JMLR* 12:2825–2830 (2011). — Ridge, k-NN, StandardScaler.

---

## TODO / lookup queue

These need a specific reference; will be filled in during the lit-search phase:

- A canonical inverse-Volterra-problem paper for model 7's motivation.
- One KAN-as-generative-component paper (or confirm we are one of the first).
- ~~One reference on perceptual / spectral losses for time-series generation~~ ✓ Engel/Yamamoto/Steinmetz added 2026-05-18 (used by B1 spectral-loss variant).
- "Generative-model fidelity in scientific simulators" — find the closest equivalent in physics ML (cosmology emulators? climate emulators?) to position our protocol.
