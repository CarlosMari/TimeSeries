import numpy as np
from scipy import integrate

def gLV(t,x_t, r, A, K):
    """
        t   -> Time step required by solve_ivp
        x_t -> Population at time t: (N,)
        r   -> Intrinsic growth rate: (N,)
        A   -> Interacion Coefficient: (N,N)
        K   -> Carrying Capacities: (N,)

        Returns:
        dx/dt -> Change in population: (N,)
    """

    return x_t * (r * (1 - x_t / K) + A @ x_t)


def generate_parameters(N=7, x0_min =0.05, x0_max = 0.2, A_min = -0.2, A_max =1):
    N = 7

    A = -np.random.uniform(A_min, A_max, (N, N))

    # Modify diagonal elements
    np.fill_diagonal(A, np.diag(A) - np.sqrt(2))

    r = np.random.uniform(0,1,N) + 0.5

    K = np.ones(N)*1000
    x0 = np.random.uniform(x0_min, x0_max, N)

    return A,r,x0,K

def generate_curves(Num_points=129):

    Nt=129
    tmax=20
    t=np.linspace(0., tmax, Nt)

    A,r,x0,K = generate_parameters()
    return integrate.solve_ivp(gLV, [0, tmax], x0, args=(r, A, K), t_eval=t).y