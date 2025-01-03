import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm.notebook import trange
import pickle
from glv_functions import *
import random


TRAIN_SEED = 74
TEST_SEED =  73
N = 7 # How many curves are there in a family
SIGMA = 0.01 # Lognormal noise variance
def generate_data(num_curves, seed, name='TRAIN'):
    np.random.seed(seed)
    sim_lists = []
    for i in trange(num_curves):
        RHO = (np.random.rand()) # correlation
        ALPHA = np.random.rand() * 5 # interaction strength

        A = elliptic_normal_matrix(N, RHO) / (np.sqrt(N) * ALPHA)  # Matrix of interactions

        # Parameters of the dynamics:
        NBR_IT = 129  # Number of iterations
        TAU = 0.093  # Time step
        x_init = np.random.uniform(0.05, 0.2, N)  # Initial condition
        r = np.random.uniform(0.2, 1, N)

        # Compute the dynamics:
        sol = custom_dynamics_LV(A, x_init, nbr_it=NBR_IT, tau=TAU, r_k=r)  # y-axis

        shape = sol.shape
        noise = np.random.lognormal(mean=0, sigma=SIGMA, size=shape)  # (7, 129)

        # Center the noise around 1
        scaling_factor = float(np.exp(SIGMA**2 / 2))
        centered_noise = noise / scaling_factor
        sol = sol * centered_noise

        # Check for NaN or extreme values
        if np.isnan(sol).any() or sol[sol > 2.0].any():
            continue


        def has_overshoot(curve):
            steady_state = np.mean(curve[-10:])
            return np.any(curve > 1.2 * steady_state)

        overshoot_count = sum(has_overshoot(curve) for curve in sol)

        # Check how many curves end at their maximum
        if overshoot_count < 3:  # Ignore families where >50% curves end at their max
            continue

        sim_lists.append(sol)

    sols = np.array(sim_lists)

    # Visualization
    fig, axs = plt.subplots(5, 5, figsize=(5, 5), sharex=True, sharey=True)
    axs = axs.flatten()

    # Choose 25 random families and plot them
    A = sols[np.random.randint(sols.shape[0], size=25), :]
    for i in range(25):
        for z in range(N):
            axs[i].plot(A[i][z])
        axs[i].axis("off")

    plt.savefig(f'./generation_comparison/{name}.png')

    # Save the filtered data
    with open(f'./data/{name}.pkl', 'wb') as output:
        pickle.dump(sols, output)

if __name__ == "__main__":
    generate_data(70000, TRAIN_SEED, 'TRAIN')
    generate_data(5000, TEST_SEED, 'TEST')