############## GR TOV SOLVER ##############
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
import pickle
import os
from ode_system import stop_at_small_r_step
from EOS import NeutronStarEOS


num_of_stars = 150
# Constants
Msun = 1.988e33
G = 6.6743e-8
c = 3e10

# Load EOS
apr_eos = NeutronStarEOS('APR')
ri = 1e-15
rf = 1.4e6

### for TOV expansion, we need gamma and beta. gamma is easily defined but beta is a mess
# import some numerical solutions for beta from mathematica and interpolate instead. works as good
data = np.genfromtxt('GR_beta_values.csv', delimiter=',')
rho_c_for_interp = data[:, 0]  
beta_values_for_interp = data[:, 1]  

# Create the interpolation function
beta_of_rho = interp1d(rho_c_for_interp, beta_values_for_interp, kind='cubic')
    
def GR_initial_conditions(beta_fn, rho_c, ri):
    bbbeta = beta_fn(rho_c)
    rho_initial = rho_c - bbbeta * ri**2
    m_initial = 4.18879 * rho_c * ri**3
    return [rho_initial, m_initial]

def GR_ode_system(r, y, P, dPdRho):
    rho, mass = y
    if rho < 1:
        return np.array([0, 0])

    expr_TOV = - (G / c**2) / r**2 * (rho * c**2 + P(rho)) * (mass + (4 * np.pi * r**3 * P(rho) / c**2)) / (1 - 2 * G * mass / (r * c**2))
    expr_TOV /= dPdRho(rho)

    drhodr = expr_TOV
    dmdr = 4 * np.pi * r**2 * rho

    return np.array([drhodr, dmdr])

def ivp_system_wrapper(r, y):
    P = apr_eos.get_pressure(extrapolate=True)
    dPdRho = apr_eos.dP_dRho()
    return GR_ode_system(r, y, P, dPdRho)

def compute_GR_TOV(beta_fn, rho_c):
    print(f"[Worker {os.getpid()}] Starting computation for rho_c = {rho_c:.3e} g/cm^3")
    initial_conditions = GR_initial_conditions(beta_fn, rho_c, ri)
    print(f"[Worker {os.getpid()}] I.Cs: rho_i = {initial_conditions[0]:0.3e} g/cm^3 and m_i = {initial_conditions[1]}")
    sol_ivp = solve_ivp(ivp_system_wrapper, [ri, rf], initial_conditions, method='RK45', rtol=1e-8, atol=1e-10, events=stop_at_small_r_step)
    print(f"[Worker {os.getpid()}] Passed by solve_ivp.")
    r_values = sol_ivp.t
    rho_values, mass_values = sol_ivp.y
    idx_outside = np.argmin(rho_values)
    radius_ns = r_values[idx_outside]
    mass_ns = mass_values[idx_outside]


    if sol_ivp.success:
        print(f"[Worker {os.getpid()}] Solution success for rho_c = {rho_c:.3e} g/cm^3")
    else:
        print(f"[Worker {os.getpid()}] Solution failed for rho_c = {rho_c:.3e} g/cm^3")

    result = {
        "mass": mass_ns / Msun,  # Convert to solar mass units
        "radius": radius_ns, 
        "rho_c": rho_c,
        "rho_values": rho_values,
        "mass_values": mass_values
    }

    return result

if __name__ == "__main__":
    num_workers = os.cpu_count()  # Or set this to a specific number you'd like to use
    print(f"Number of workers being used: {num_workers}")
    
    rho_c_values = [10**i for i in np.linspace(np.log10(4e14), np.log10(1.778e16), num_of_stars)]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_GR_TOV, [beta_of_rho] * len(rho_c_values), rho_c_values))

    # Save results
    with open("./output/GR/GR_TOV_solutions.pkl", 'wb') as file:
        pickle.dump(results, file)
    print("Saved GR TOV solutions to './output/GR_TOV_solutions.pkl'")