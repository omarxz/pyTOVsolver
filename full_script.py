
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import minimize
from EOS import NeutronStarEOS
from ode_system import *
from omars_little_helpers import *
# from concurrent.futures import ProcessPoolExecutor #to parallize computations

apr_eos = NeutronStarEOS('APR')

# define the domain for r: we start away from r=0 to avoid the TOV singularity
r_center = 1e-15
# slighly above all the expected radii. early termination will occur regardless
r_rad = 2e6 
############### let's write the script for only 1 star first ###############
rho_c = 10**15.7
# let's use the minimum of the potetial as our initial guess 
a_minimized = axion_initial_guess(rho_c)

# obtain the boundary conditions at r = 1e-15 cm by expanding the equations linearly
# [a_initial_guess, a_prime_initial, nu_initial, llambda_initial, rho_initial]
initial_conditions = create_boundary_conditions(eos_class=apr_eos,
rho_c=rho_c,
nu_c=1,
lambda_c=0,
a_c=a_minimized,
ri=r_center)

# wrap the initial solver to use the interpolated EoS
def inside_ivp_wrapper(r, y):
    P = apr_eos.get_pressure(extrapolate=True)
    dPdRho = apr_eos.dP_dRho()
    return inside_ivp_system(r, y, P, dPdRho)

# IVP solver
def solve_interior(initial_conditions):
    print("     Solving the interior....")

    # Solve the ivp
    sol = solve_ivp(inside_ivp_wrapper, [r_center,r_rad], initial_conditions, method='LSODA', events = stop_at_small_r_step)

      # Extract boundary conditions at R from the interior solution
    a_in = sol.y[0, :]  # a at the surface
    a_prime_in = sol.y[1, :]  # a' at the surface
    nu_in = sol.y[2, :]
    llambda_in = sol.y[3, :]
    rho = sol.y[4, :]
    radius = sol.t_events[0][0]
    r_inside = sol.t
    # Process the solution
    # Check if the solution is successful and process it
    if sol.success:
        print("     IVP Solution found!")
        print(f"    ---------------------------------------------\n")
        print(sol.message)
    else:
        print("     Solution was not successful.")
        print(f"    ---------------------------------------------\n")
        print(sol.message)
    return sol, a_in, a_prime_in, nu_in, llambda_in, rho, radius

def solve_bvp_outside(r_outside, y_initial, outside_bc_func):
    print("     Solving the exterior....")
    sol = solve_bvp(outside_bvp_system, outside_bc_func, r_outside, y_initial, max_nodes=1000000, tol=2.22e-14)

    # Process the solution
    a_out = sol.sol(r_outside)[0]
    a_prime_out = sol.sol(r_outside)[1]
    nu_out = sol.sol(r_outside)[2]
    llambda_out = sol.sol(r_outside)[3]

    # Check if the solution is successful and process it
    if sol.success:
        print('     BVP Solution found!')
        print(f"    ---------------------------------------------\n")
    else:
        print("     Solution was not successful.")
        print(f"    ---------------------------------------------\n")

    return sol, a_out, a_prime_out, nu_out, llambda_out

optimized_sol = {}
def continuity_cost(a_initial_guess):
    global optimized_sol
    # Step 1: Solve the interior problem with the current guess for a_initial
    a_initial_guess = a_initial_guess[0]
    print(f"Initial guess: {a_initial_guess}")

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
    sol_exterior, a_out, a_prime_out, nu_out, llambda_out = solve_bvp_outside(r_outside, y_initial, outside_bc_func)
    if not sol_exterior.success:
        return np.inf  # Penalize failed solutions heavily
    mass = c**2 * r_outside[-1] / (2*G) * (1. - np.exp(-llambda_out[-1]))/Msun

    optimized_sol = {
    "a_c": a_initial_guess, "radius": radius, "mass": mass, "rho": rho, "a_in": a_in, "a_out": a_out, "a_prime_in": a_prime_in, "a_prime_out": a_prime_out,
    "nu_in": nu_in, "nu_out": nu_out, "llambda_in": llambda_in, "llambda_out": llambda_out, 
    "r_inside": sol_interior.t, "r_outside": r_outside
    }
    # Compute the cost: Here, we aim for a smooth transition, so ideally, a_out[0] - a_R should be close to 0
    # and a_prime_out[0] - a_prime_R should be close to 0. Adjust the cost function as needed.
    cost_a = (a_out[0] - a_R)**2   # Simple squared difference
    cost_a_prime = (a_prime_out[0] - a_prime_R)**2
    print(f"Cost for a_R = {cost_a:0.3e}")
    print(f"Cost for a_prime_R = {cost_a:0.3e}")
    cost = cost_a+cost_a_prime
    
    return cost 

# Initial guess for a_initial
a_initial_guess = initial_conditions[0]  # Use the a_initial you calculated earlier as the starting point

# Perform the optimization
result = minimize(continuity_cost, a_initial_guess, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-9})

if result.success:
    optimized_a_initial = result.x[0]
    print(f"Optimized a_initial: {optimized_a_initial}")
else:
    print("Optimization was not successful. Try adjusting the initial guess or method.")
# Initial guess for a_initial
a_initial_guess = initial_conditions[0]  # Use the a_initial you calculated earlier as the starting point

# Perform the optimization
result = minimize(continuity_cost, a_initial_guess, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-9})

if result.success:
    optimized_a_initial = result.x[0]
    print(f"Optimized a_initial: {optimized_a_initial}")
else:
    print("Optimization was not successful. Try adjusting the initial guess or method.")

# Step 1: Solve the interior problem with the current guess for a_initial

