import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
import pickle
import os

def objective_function(x, c):
    """Objective function (x - c)^3."""
    return (x - c) ** 2

def optimize_for_c(c):
    """Minimize the objective function for a given c."""
    # Initial guess for x
    x0 = np.array([0.0])
    # Minimize the objective function
    res = minimize(objective_function, x0, args=(c,), method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-9})
    if res.success:
        optimized_x = res.x[0]
        print(f"[{os.getpid()}] Optimization successful for c = {c}, optimized x = {optimized_x}")
        return {'c': c, 'optimized_x': optimized_x}
    else:
        print(f"[{os.getpid()}] Optimization failed for c = {c}")
        return None

if __name__ == "__main__":
    # List of c values to optimize for
    c_values = [i for i in np.linspace(2, 4, 10)]
    # Total number of optimizations to perform
    total_optimizations = len(c_values)

    # Use ProcessPoolExecutor to parallelize the optimization
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(optimize_for_c, c) for c in c_values]
        results = [future.result() for future in futures if future.result() is not None]

    # Save to a pickle file
    pickle_file_path = 'test_optimization_results.pkl'
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(results, file)

    print(f"Saved optimization results to {pickle_file_path}")