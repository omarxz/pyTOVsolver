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

# Constants
Msun = 1.988e33
G = 6.6743e-8
c = 3e10

# Load EOS
apr_eos = NeutronStarEOS('APR')
ri = 1e-15
rf = 2e6

def GR_initial_conditions(eos_class, rho_c, ri):
    P = eos_class.get_pressure(extrapolate=True)
    dPdrho = eos_class.dP_dRho()
    rho_initial = rho_c - 228.917 * ri**2
    m_initial = 4.18879e14 * ri**3
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

def compute_GR_TOV(rho_c):
    print(f"[Worker {os.getpid()}] Starting computation for rho_c = {rho_c:.3e} g/cm^3")
    initial_conditions = GR_initial_conditions(apr_eos, rho_c, ri)
    sol_ivp = solve_ivp(ivp_system_wrapper, (ri, rf), initial_conditions, method='RK45', events=stop_at_small_r_step)
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
        "radius": radius_ns / 1e5,  # Convert to km
        "rho_c": rho_c,
        "rho_values": rho_values,
        "mass_values": mass_values
    }

    return result

if __name__ == "__main__":
    num_workers = os.cpu_count()  # Or set this to a specific number you'd like to use
    print(f"Number of workers being used: {num_workers}")
    
    rho_c_values = [10**i for i in np.linspace(14.3, 16.25, 100)]  # Example range of rho_c
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_GR_TOV, rho_c_values))

    # Save results
    with open('GR_TOV_solutions.pkl', 'wb') as file:
        pickle.dump(results, file)
    print("Saved GR TOV solutions to 'GR_TOV_solutions.pkl'")