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


def generate_parameters(N=7, x0_min =0.05, x0_max = 0.1, A_min = -1, A_max =0.2):
    A = np.random.uniform(A_min, A_max, (N, N)) # normal 0, std = 1
    np.fill_diagonal(A, np.diag(A) - np.sqrt(2)) # Multiplicar 1.1
    r = np.random.lognormal(0, 0.5, N)
    K = np.ones(N)*1000
    x0 = np.random.uniform(x0_min, x0_max, N)

    return A,r,x0,K

def generate_curves(Num_points=129):

    Nt=129
    tmax=20
    t=np.linspace(0., tmax, Nt)

    A,r,x0,K = generate_parameters()
    return integrate.solve_ivp(gLV, [0, tmax], x0, args=(r, A, K), t_eval=t).y


def generate_curves_Mario(myseed=0, noise_level=0.01, species=3, tmax=0, n_points=25,
               folder=".", plot_this=True, parameters={}):
    
    def lotka_volterra(t, x, params):
        K = len(x)
        r = params[:K]
        b = np.array(params[K:]).reshape((K, K)).T
        dxdt = np.zeros(K)
        for i in range(K):
            dxdt[i] = r[i] * x[i]
            for j in range(K):
                dxdt[i] += b[i, j] * x[i] * x[j]
        return dxdt

    if myseed == 0:
        myseed = np.random.randint(1, 100001)
    np.random.seed(myseed)

    if len(parameters) == 0:
        var_names = []
        for i in range(1, species + 1):
            var_names.append(f"r{i}")
        for i in range(1, species + 1):
            for j in range(1, species + 1):
                var_names.append(f"a{i}{j}")

        flag = False
        while not flag:
            r0 = np.random.exponential(scale=2.0, size=species)  # rexp(rate=0.5)
            a0 = np.random.randn(species, species)
            for i in range(species):
                a0[i, i] = -np.random.exponential(scale=2.0)

            params = np.concatenate([r0, a0.T.flatten()])
            xss = np.linalg.solve(a0, -r0)
            d0 = np.diag(xss)
            eigenvalues = np.real(np.linalg.eigvals(d0 @ a0))

            flag = np.all(eigenvalues <= 0) and np.all(xss > 0)
            #print(flag)

        #print("xss =", xss)
        #print("eig =", eigenvalues)

        initial_condition = np.random.exponential(scale=0.1, size=species)
    else:
        #print("Species:", species)
        species = parameters['K']
        #print("Species:", species)
        myseed = parameters['myseed']
        params = parameters['params']
        initial_condition = parameters['initial_condition']
        r0 = params[:species]
        a0 = np.array(params[species:]).reshape((species, species)).T
        xss = np.linalg.solve(a0, -r0)
        d0 = np.diag(xss)
        eigenvalues = np.real(np.linalg.eigvals(d0 @ a0))
        #print(eigenvalues)

    # Time points
    if tmax == 0:
        tmax = -20 / np.min(eigenvalues)
    times = np.linspace(0, tmax, n_points)

    # Solve ODE
    sol = integrate.solve_ivp(fun=lambda t, y: lotka_volterra(t, y, params),
                    t_span=(0, tmax),
                    y0=initial_condition,
                    t_eval=times,
                    method='RK45')

    out = sol.y
    #print(f'{out.shape}')
    # Add noise
    #for i in range(species):
    #    out[:, i] = np.random.lognormal(mean=np.log(1e-6 + np.abs(out[:, i])), sigma=noise_level)

    return out
    
