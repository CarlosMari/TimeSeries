import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
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
    num_sols = 0 
    pbar = tqdm(range(num_curves), desc='Finding Correct Solutions')
    scaling_factor = float(np.exp(SIGMA**2 / 2))

    for i in pbar:
        RHO = (np.random.rand()) # correlation
        ALPHA = np.random.rand() * 5 # interaction strength

        A = elliptic_normal_matrix(N, RHO) / (np.sqrt(N) * ALPHA)  # Matrix of interactions

        # Parameters of the dynamics:
        NBR_IT = 129  # Number of iterations
        TAU = 0.093  # Time step
        x_init = np.random.uniform(0.02, 0.1, N)  # Initial condition
        r = np.random.uniform(0.2, 1, N)

        # Compute the dynamics:
        sol = custom_dynamics_LV(A, x_init, nbr_it=NBR_IT, tau=TAU, r_k=r)  # y-axis

        shape = sol.shape
        noise = np.random.lognormal(mean=0, sigma=SIGMA, size=shape)  # (7, 129)

        # Center the noise around 1
        centered_noise = noise / scaling_factor
        sol = sol * centered_noise

        # Check for NaN or extreme values
        if np.isnan(sol).any() or sol[sol > 2.0].any():
            continue

        steady_states = np.mean(sol[:, -10:], axis=1)
        overshoot_flags = np.max(sol, axis=1) > 1.2 * steady_states
        extinction_flags = steady_states <= sol[:, 0]

        overshoot_count = np.sum(overshoot_flags)
        extinct_count = np.sum(extinction_flags)

        # Check how many curves end at their maximum // how many go extinct
        if overshoot_count < 3 or extinct_count > 1:  # Ignore families 
            continue

        num_sols += 1
        pbar.set_postfix({'Correct Solutions': num_sols})
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
    print(f'Generated data {sols.shape}')
    # Save the filtered data
    with open(f'./data/{name}.pkl', 'wb') as output:
        pickle.dump(sols, output)

if __name__ == "__main__":
    generate_data(2500000, TRAIN_SEED, 'TRAIN')
    generate_data(50000, TEST_SEED, 'TEST')