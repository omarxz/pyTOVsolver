import re
import os
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import minimize
from EOS import NeutronStarEOS
from ode_system import *
import ode_system
from omars_little_helpers import *
from concurrent.futures import ProcessPoolExecutor #to parallize computations
import warnings

# extract the solved stars from the failed putput
pattern = r"Found a_c = ([-+]?\d*\.?\d+) for rho_c = ([-+]?\d*\.?\d+e[+]\d*)"

# Initialize empty lists to store a_c and rho_c values
a_c_values = []
rho_c_values = []

g_s_N = 1e-21
ode_system.mu = (NeutronMass)/(g_s_N) * PhiFaGeVToCGs
ode_system.ma = 2.8e-12 * 1e-9 * cmToGeVInv
ode_system.fa = 1e15 * PhiFaGeVToCGs

# Read the log file
pre_name = "ma12_gsN1e-21"
path = f"/home/oramadan/Axions/pyTOVsolver/output/{pre_name}"
with open(f"{path}/output_{pre_name}.log", "r") as file:
    # Iterate through each line in the file
    for line in file:
        # Use regex to search for the pattern in the line
        match = re.search(pattern, line)
        # If a match is found, extract the values of a_c and rho_c
        if match:
            a_c = float(match.group(1))
            rho_c = float(match.group(2))
            # Append the values to the respective lists
            a_c_values.append(a_c)
            rho_c_values.append(rho_c)

# Convert lists to numpy arrays
a_c_array = np.array(a_c_values)
rho_c_array = np.array(rho_c_values)

print(f"[{os.getpid()}] Number of stars found detected: {len(rho_c_array)}\n")
# Ignore specific SciPy UserWarnings regarding tolerance levels
warnings.filterwarnings("ignore", message="`tol` is too low, setting to 2.22e-14")

# Detect the number of cores/processors
num_cores = os.cpu_count()

print(f"Number of cores detected: {num_cores}\n")

# load eos, it's the same for all workers so we can keep it global
apr_eos = NeutronStarEOS('APR')


# define the domain for r: we start away from r=0 to avoid the TOV singularity
r_center = 1e-15
# slighly above all the expected radii. early termination will occur regardless
r_rad = 3e6 

# wrap the initial solver to use the interpolated EoS
def inside_ivp_wrapper(r, y):
    P = apr_eos.get_pressure(extrapolate=True)
    dPdRho = apr_eos.dP_dRho()
    return inside_ivp_system(r, y, P, dPdRho)

# IVP solver
def solve_interior(initial_conditions):
    print(f"[{os.getpid()}] Solving the interior....")
    # Solve the ivp
    sol = solve_ivp(inside_ivp_wrapper, [r_center,r_rad], initial_conditions, method='LSODA', events = stop_at_small_r_step)

    # Extract boundary conditions at R from the interior solution
    a_in = sol.y[0, :]  # a at the surface
    a_prime_in = sol.y[1, :]  # a' at the surface
    nu_in = sol.y[2, :]
    llambda_in = sol.y[3, :]
    rho = sol.y[4, :]
    r_inside = sol.t

    # Check if the solution is successful and process it
    if sol.success:
        print(f"[{os.getpid()}] IVP Solution found!")
        print("     ", sol.message)
        if sol.t_events[0].size > 0:
            event_message = sol.t_events[0][0]  # Get the message of the triggered event
            print(f"[{os.getpid()}] Event triggered: {event_message/1e5:0.4} km")
            radius = sol.t_events[0][0]
    else:
        print(f"[{os.getpid()}] Warning: No events were triggered. Setting default radius value.")
        idx_outside = np.argmin(rho)
        print(f"[{os.getpid()}] Radius detected at rho = {rho[idx_outside]} g/cm^3 and R = {r_inside[idx_outside]/1e5} km")
        radius = r_inside[idx_outside]
    return sol, a_in, a_prime_in, nu_in, llambda_in, rho, radius

def solve_bvp_outside(r_outside, y_initial, outside_bc_func, initial_tol):
    print(f"[{os.getpid()}] Solving the exterior....")
    tol = initial_tol  # Initialize the tolerance
    while tol<=1.:
        sol = solve_bvp(outside_bvp_system, outside_bc_func, r_outside, y_initial, max_nodes=100000, tol=tol)
        # store the solution
        a_out = sol.sol(r_outside)[0]
        a_prime_out = sol.sol(r_outside)[1]
        nu_out = sol.sol(r_outside)[2]
        llambda_out = sol.sol(r_outside)[3]
    
        # Check if the solution is successful
        if sol.success:
            print(f"[{os.getpid()}] BVP Solution found with tol = {tol}!")
            return sol, a_out, a_prime_out, nu_out, llambda_out
        else:
            print(f"[{os.getpid()}] Solution was not successful with tolerance {tol}. Retrying with higher tolerance.")
            tol *= 10  # Increase the tolerance by an order of magnitude
    print(f"[{os.getpid()}] Exceeded maximum tolerance. Solution not found.")
    return None, None, None, None, None


def full_solve(a_c, rho_c):
    # Initial tolerance value
    initial_tol = 1e-14
    initial_conditions = create_boundary_conditions(eos_class=apr_eos,
    rho_c=rho_c,
    nu_c=1,
    lambda_c=0,
    a_c=a_c,
    ri=r_center 
    )
    sol_interior, a_in, a_prime_in, nu_in, llambda_in, rho, radius = solve_interior(initial_conditions)

    # Extract boundary conditions at R from the interior solution
    a_R = a_in[-1]  # a at the surface
    a_prime_R = a_prime_in[-1]  # a' at the surface
    nu_R = nu_in[-1]
    llambda_R = llambda_in[-1]

    # mesh for outside solution
    r_far = 2e8
    r_outside = create_r_mesh(radius,r_far,4000)
    # Update the boundary conditions for the exterior problem
    outside_bc_func = create_outside_bc(a_R, a_prime_R, nu_R, llambda_R)

    # gues initial solution
    y_initial = np.zeros((4, r_outside.size))  # Initialize the array with zeros
    y_initial[0, :] = np.linspace(a_R, 0, len(r_outside))
    # linearly interpolate the remaining values of a(r) from ac to final_value_a
    y_initial[1, :] = np.linspace(a_prime_R, 0, len(r_outside))  # a_prime(r)
    y_initial[2, :] = nu_R  # nu(r)
    y_initial[3, :] = llambda_R  # llambda(r)
    # Step 2: Solve the exterior problem with these boundary conditions
    sol_exterior, a_out, a_prime_out, nu_out, llambda_out = solve_bvp_outside(r_outside, y_initial, outside_bc_func, initial_tol)
    mass = c**2 * r_outside[-1] / (2*G) * (1. - np.exp(-llambda_out[-1]))/Msun

    results = {
    "a_c": a_c, "radius": radius, "mass": mass, "rho": rho, "a_in": a_in, "a_out": a_out, "a_prime_in": a_prime_in*1e5, "a_prime_out": a_prime_out*1e5,
    "nu_in": nu_in, "nu_out": nu_out, "llambda_in": llambda_in, "llambda_out": llambda_out, 
    "r_inside": sol_interior.t, "r_outside": r_outside
    }

    return results


path_to_pickles = f"{path}/{pre_name}_full_star_solutions.pkl"
if __name__ == "__main__":
    # Use ProcessPoolExecutor to parallelize the optimization
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(full_solve, a_c, rho_c) for a_c, rho_c in zip(a_c_array, rho_c_array)]
        results = [future.result() for future in futures if future.result() is not None]

        # Load existing results from the pickle file
        
with open(path_to_pickles, 'wb') as file:
    pickle.dump(results, file)
    print(f"[{os.getpid()}] Saved the star solutions.")
    
print(f"[{os.getpid()}] Optimization is complete and solutions have been saved.")
print(f"Calculated all {len(a_c_array)} stars and saved. Wohooooo!")

