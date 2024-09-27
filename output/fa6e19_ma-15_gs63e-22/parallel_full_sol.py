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

###########################################################################
# save a copy of the current script since it will be modified
pre_name = "fa6e19_ma-15_gs63e-22"
output_dir = "./output/" + pre_name
print(f"Working directiory from .py is {output_dir}")
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Path to the current script
script_path = __file__
# Name of the script file
script_name = os.path.basename(script_path)
# Define the path for the copy
copy_path = os.path.join(output_dir, script_name)

# Read the current script and write its content to the new location
with open(script_path, 'r') as original_file:
    script_content = original_file.read()

with open(copy_path, 'w') as copy_file:
    copy_file.write(script_content)

print(f"Saved a copy of the script to {copy_path}")
######################### uncomment and modify to change the params #########################
g_s_N = 5e-22
ode_system.mu = (NeutronMass)/(g_s_N) * PhiFaGeVToCGs
ode_system.ma = 1e-15 * 1e-9 * cmToGeVInv
ode_system.fa = 2e19 * PhiFaGeVToCGs

# Ignore specific SciPy UserWarnings regarding tolerance levels
warnings.filterwarnings("ignore", message="`tol` is too low, setting to 2.22e-14")

# Detect the number of cores/processors
num_cores = os.cpu_count()

num_of_stars = 80
print(f"[{os.getpid()}] Number of cores detected: {num_cores}\n")
print(f"Parallizing {num_of_stars} stars")

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

def continuity_cost(a_initial_guess, rho_c):
    initial_tol = 1e-14
    # Step 1: Solve the interior problem with the current guess for a_initial
    if not np.isscalar(a_initial_guess):
        a_initial_guess = a_initial_guess[0]
    print(f"[{os.getpid()}] Updated guess: {a_initial_guess}")

    initial_conditions = create_boundary_conditions(eos_class=apr_eos,
        rho_c=rho_c,
        nu_c=1,
        lambda_c=0,
        a_c=a_initial_guess,
        ri=r_center 
    )
    sol_interior, a_in, a_prime_in, nu_in, llambda_in, rho, radius = solve_interior(initial_conditions)
    if not sol_interior.success:
        return np.inf  # Penalize failed solutions heavily
    
    # Extract boundary conditions at R from the interior solution
    a_R = a_in[-1]  # a at the surface
    a_prime_R = a_prime_in[-1]  # a' at the surface
    nu_R = nu_in[-1]
    llambda_R = llambda_in[-1]

    # create the mesh for the BVP outside
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
    if not sol_exterior.success:
        return np.inf  # Penalize failed solutions heavily
    mass = c**2 * r_outside[-1] / (2*G) * (1. - np.exp(-llambda_out[-1]))/Msun

    # Cost term ensuring a_prime_out[-1] is close to zero
    cost_a_prime = a_prime_out[-1]**2  # Squared difference from zero
    print(f"[{os.getpid()}] MSE cost for a_prime(r_far) = {cost_a_prime:0.4e}")
    
    # Cost term ensuring a_R and a_out[0] are negative and identical
    if a_out[0] >= 0 or a_R >= 0 or a_in[0] == a_R:
        print(f"[{os.getpid()}] a_out[0] >= 0 or a_R >= 0 or a_in[0] == a_R. Cost = np.inf")
        return  np.inf
    else:
        cost_a_R = abs(a_R - a_out[0])
        print(f"[{os.getpid()}] MSE cost for a(R) = {cost_a_R:0.4e}")
        if cost_a_R > 1:
            print(f"[{os.getpid()}] MSE cost_a_R > 1. Return np.inf")
            return np.inf

    # Calculate magnitudes of the costs
    magnitude_a_R = np.abs(cost_a_R)
    magnitude_a_prime = np.abs(cost_a_prime)

    # Calculate relative magnitudes
    total_magnitude = magnitude_a_R + magnitude_a_prime
    relative_magnitude_a_R = magnitude_a_R / total_magnitude
    relative_magnitude_a_prime = magnitude_a_prime / total_magnitude
    
    # Choose epsilon
    base_epsilon = 1e-10  # Adjust this value as needed based on your precision requirements
    epsilon_scale = max(relative_magnitude_a_R, relative_magnitude_a_prime)
    epsilon = base_epsilon * epsilon_scale
    
    # Apply epsilon in your calculations
    normalized_cost_a_R = cost_a_R / (magnitude_a_R + epsilon)
    normalized_cost_a_prime = cost_a_prime / (magnitude_a_prime + epsilon)

    # Combine the normalized costs with equal weights (50/50)
    cost = 0.5 * normalized_cost_a_R + 0.5 * normalized_cost_a_prime
    print(f"[{os.getpid()}] Normalized total cost = {cost}")
    return cost
    
    


def compute_for_rho_c(rho_c, g_s_N=g_s_N):
    print(f"[{os.getpid()}] Optimizing for rho_c = {rho_c:0.3e} g/cm^(3)")
    a_minimized = axion_initial_guess(rho_c, g_s_N)
    # Wrap the call to continuity_cost so it returns only the cost to minimize
    result = minimize(continuity_cost, a_minimized, args=(rho_c,), method='Nelder-Mead', options={'maxiter': 250, 'xatol': 1e-9}) # OR maybe 150 is too much? default is 100
    if result.success: 
        print(f"[{os.getpid()}] Found a_c = {result.x[0]} for rho_c = {rho_c:e} g/cm^(3)")
        return {'rho_c': rho_c, 'a_c': result.x[0]}
    else:
        print(f"[{os.getpid()}] Optimization failed for rho_c = {rho_c:e}")
        return None

if __name__ == "__main__":
    rho_c_values = [10**i for i in np.linspace(np.log10(4e14), np.log10(1.778e16), num_of_stars)]
    # Use ProcessPoolExecutor to parallelize the optimization
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_for_rho_c, rho_c) for rho_c in rho_c_values]
        results = [future.result() for future in futures if future.result() is not None]
    # Save to a pickle file
    pickle_file_path = f"{output_dir}/{pre_name}_rho_ac.pkl"
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(results, file)
    print(f"Saved dict of rho_c and a_c to {pickle_file_path}")

print("Optimization is complete. Now, let's calculate the full solution. Wohooooo!")

# Load the contents of the pickle file
with open(pickle_file_path, 'rb') as file:
    optimization_results = pickle.load(file)

# Define a wrapper function that takes a single argument, for the executor.map method
def compute_full_solution(result):
    rho_c = result['rho_c']
    a_c = result['a_c']
    
    # Compute the full solution for the given rho_c and a_c
    full_solution = full_solve(a_c, rho_c)
    return full_solution

print("Calculating full solutions in parallel...........")

# Use ProcessPoolExecutor to parallelize the full solution computation
with ProcessPoolExecutor() as executor:
    # Map each optimization result to the compute_full_solution function
    full_solutions = list(executor.map(compute_full_solution, optimization_results))

# Save the full solutions to a new pickle file for further analysis
path_to_final_results = f"{output_dir}/{pre_name}_full_star_solutions.pkl"
with open(path_to_final_results, 'wb') as file:
    pickle.dump(full_solutions, file)

print(f"Calculated full solutions for {len(full_solutions)} stars and saved to {path_to_final_results}")