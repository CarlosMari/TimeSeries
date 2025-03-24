import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import pickle
#from glv_functions import *
import random
from custom_glv import generate_curves

LOG = False

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

        # Compute the dynamics:
        sol = generate_curves() 

        shape = sol.shape
        noise = np.random.lognormal(mean=0, sigma=SIGMA, size=shape)  # (7, 129)

        # Center the noise around 1
        centered_noise = noise / scaling_factor
        sol = sol * centered_noise


        steady_states = np.mean(sol[:, -10:], axis=1)
        overshoot_flags = np.max(sol, axis=1) > 1.2 * steady_states
        overshoot_count = np.sum(overshoot_flags)

        # Check for NaN or extreme values
        if np.isnan(sol).any() or sol[sol > 1.0].any() or np.any(np.max(sol, axis=1) < 0.1) or overshoot_count < 3:
            continue

        if LOG:
            print('=================================')
            print(f'Correct Family {i}')
            for curve in sol:
                print(f'Max {np.max(curve)}')
        
            print('=================================')

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
    generate_data(5000000, TRAIN_SEED, 'TRAIN_NEw')
    generate_data(400000, TEST_SEED, 'TEST_NEW')