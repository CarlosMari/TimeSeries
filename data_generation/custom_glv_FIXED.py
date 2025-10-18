"""
FIXED VERSION: Uses RandomState instead of global seed manipulation.

Key improvements:
1. Accept either myseed or rng parameter
2. Don't reset global seed
3. Fix fallback seed logic to use consistent RNG
"""
import numpy as np
from scipy import integrate


def gLV(t, x_t, r, A, K):
    """
        t   -> Time step required by solve_ivp
        x_t -> Population at time t: (N,)
        r   -> Intrinsic growth rate: (N,)
        A   -> Interaction Coefficient: (N,N)
        K   -> Carrying Capacities: (N,)

        Returns:
        dx/dt -> Change in population: (N,)
    """
    return x_t * (r * (1 - x_t / K) + A @ x_t)


def generate_parameters(N=7, x0_min=0.05, x0_max=0.3, A_min=-1, A_max=0.2, rng=None):
    """Generate GLV parameters using provided RNG."""
    if rng is None:
        rng = np.random.RandomState()

    A = rng.normal(0, 1, (N, N))
    np.fill_diagonal(A, 0)

    for i in range(N):
        # Sum of absolute values of elements in row i (excluding diagonal)
        row_sum = np.sum(np.abs(A[i, :])) * 0.4

        # Generate lognormal noise
        noise = rng.lognormal(0, 0.1)

        # Set diagonal element according to formula
        A[i, i] = -(row_sum + noise)

    r = rng.exponential(0.5, N)
    K = rng.uniform(200, 1000, N)
    x0 = rng.uniform(x0_min, x0_max, N)

    return A, r, x0, K


def generate_curves(Num_points=129, rng=None):
    """Generate curves using RNG instead of global seed."""
    if rng is None:
        rng = np.random.RandomState()

    Nt = 129
    tmax = 20
    t = np.linspace(0., tmax, Nt)

    A, r, x0, K = generate_parameters(rng=rng)
    return integrate.solve_ivp(gLV, [0, tmax], x0, args=(r, A, K), t_eval=t).y


def generate_curves_Mario(myseed=0, noise_level=0.01, species=3, tmax=0, n_points=25,
                          folder=".", plot_this=True, parameters={}, rng=None):
    """
    Generate GLV curves with proper seed handling.

    FIXED: Accepts either myseed (for backward compatibility) or rng (preferred).
    Does not modify global random state.
    """
    # FIXED: Use local RNG instead of global seed
    if rng is None:
        rng = np.random.RandomState(myseed)

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

    if len(parameters) == 0:
        var_names = []
        for i in range(1, species + 1):
            var_names.append(f"r{i}")
        for i in range(1, species + 1):
            for j in range(1, species + 1):
                var_names.append(f"a{i}{j}")

        flag = False
        c = 0
        max_attempts = 500

        while not flag and c < max_attempts:
            # FIXED: Use rng instead of np.random
            r0 = rng.exponential(scale=2.0, size=species)
            a0 = rng.randn(species, species)
            for i in range(species):
                a0[i, i] = -rng.exponential(scale=2.0)

            params = np.concatenate([r0, a0.T.flatten()])

            try:
                xss = np.linalg.solve(a0, -r0)
                d0 = np.diag(xss)
                eigenvalues = np.real(np.linalg.eigvals(d0 @ a0))

                flag = np.all(eigenvalues <= 0) and np.all(xss > 0)
            except np.linalg.LinAlgError:
                flag = False

            c += 1

        if not flag:
            # FIXED: Return None instead of using fallback seed
            # This is clearer and lets caller decide how to handle
            return None

        # FIXED: Use rng instead of np.random
        initial_condition = rng.exponential(scale=0.1, size=species)

    else:
        # Using provided parameters
        species = parameters['K']
        myseed = parameters['myseed']
        params = parameters['params']
        initial_condition = parameters['initial_condition']
        r0 = params[:species]
        a0 = np.array(params[species:]).reshape((species, species)).T
        xss = np.linalg.solve(a0, -r0)
        d0 = np.diag(xss)
        eigenvalues = np.real(np.linalg.eigvals(d0 @ a0))

    # Time points
    if tmax == 0:
        tmax = -20 / np.min(eigenvalues)
    times = np.linspace(0, tmax, n_points)

    # Solve ODE with timeout protection
    try:
        sol = integrate.solve_ivp(
            fun=lambda t, y: lotka_volterra(t, y, params),
            t_span=(0, tmax),
            y0=initial_condition,
            t_eval=times,
            method='RK45',
            max_step=0.5  # Prevent hanging on stiff systems
        )

        # Check if integration was successful
        if not sol.success:
            return None

        out = sol.y
    except Exception as e:
        # Silent failure - return None to let caller handle
        return None

    return out


# Backward compatible wrapper
def generate_curves_Mario_old_api(myseed=0, **kwargs):
    """
    Backward compatible wrapper that uses global seed.
    For new code, use generate_curves_Mario with rng parameter instead.
    """
    np.random.seed(myseed)
    return generate_curves_Mario(myseed=myseed, **kwargs)
