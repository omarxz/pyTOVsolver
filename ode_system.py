import numpy as np


gToGeV = 5.62e23 # (g/GeV)*)
cmInvToGeV = 1.98e-14 # (*cm^-1/GeV*) 
cmToGeVInv = 1/cmInvToGeV # (*GeV^-1/cm*)
gCm3ToGeV4 = (cmInvToGeV)**3 * gToGeV # (*g/cm^3 -> GeV^4*)
GeV4TogCm3 = 1/gCm3ToGeV4 # (*GeV^4 -> g/cm^3*)
sInvtoGeV = 6.58e-25 # (*GeV.s*)
GeVToErg = 1/(6.24e2) # (*Erg/GeV*)
PressueToGeV4 = gToGeV * cmInvToGeV * sInvtoGeV**2 # lol 
PhiFaGeVToCGs =( 1/gToGeV * 1/cmToGeVInv * (1/sInvtoGeV)**2)**(1/2) # (*(GeV . GeV^-1 . GeV^2)^(1/2) -> (g . cm . s^-2)*)
Msun = 1.988e33 # grams
G = 6.6743e-8 # (*dyne cm^2/g^2*)
c = 3e10 # (*cm/s*) 
fa = 1e15*PhiFaGeVToCGs # (*g^1/2 cm^1/2 s^-2*)
ma = 1e-11*1e-9 * cmToGeVInv; # (*1/L*)
NeutronMass = 0.939565 # (*GeV*) 
mu =(NeutronMass)/(1e-25) * PhiFaGeVToCGs # (*GeV*)
#ri = 1e-15
#rf = 10^8

def create_boundary_conditions(eos_class, rho_c, nu_c, lambda_c, a_c):
    """Create a boundary conditions function with specific initial conditions."""
    
    P = eos_class.get_pressure()
    dPdrho = eos_class.dP_drho(rho_c) # need to make this method
    nu_initial = nu_c + (4 * G * np.pi * ri**2 * \
                            (-2 * fa**2 * ma**2 * mu - 2 * a_c * c**2 * fa * rho_c + \
                            c**2 * mu * rho_c + 2 * fa**2 * ma**2 * mu * np.cos(a_c) + \
                                3 * mu * P(rho_c))) / (3 * c**4 * mu)
    
    llambda_initial = lambda_c + (8 * G * np.pi * ri**2 * \
                 (fa**2 * ma**2 * mu + a_c * c**2 * fa * rho_c + c**2 * mu * rho_c -  \
                  fa**2 * ma**2 * mu * np.cos(a_c))) / (3 * c**4 * mu)
    
    rho_initial = rho_c - 1/(6 * mu * dPdrho) * ri**2 * (
        (4 * G * np.pi * rho_c * \
         (-2 * fa**2 * ma**2 * mu - 2 * a_c * c**2 * fa * rho_c + c**2 * mu * rho_c + 2 * fa**2 * ma**2 * mu * np.cos(a_c) + 3 * mu * P(rho_c))) \
            / c**2 + (4 * G * np.pi * P(rho_c) * (-2 * fa**2 * ma**2 * mu - 2 * a_c * c**2 * fa * rho_c + \
                                           c**2 * mu * rho_c + 2 * fa**2 * ma**2 * mu * np.cos(a_c) + 3 * mu * P(rho_c))) / c**4 \
                                            + (c**2 * rho_c * (c**2 * rho_c + fa * ma**2 * mu * np.sin(a_c))) / mu - \
                                                  (3 * P(rho_c) * (c**2 * rho_c + fa * ma**2 * mu * np.sin(a_c))) / mu )
    a_initial_guess = a_c + (ri**2 * (c**2 * rho_c + fa * ma**2 * mu * np.sin(a_c))) / (6 * fa * mu)

    def boundary_conditions(ya, yb):
        return np.array([
            ya[1],                 #  ya[1] a_prime(ri) = 0
            yb[1],                 # a_prime(rf) = 0
            ya[2] - nu_initial,    # nu(ri) = nu_initial
            ya[3] - llambda_initial, # lambda(ri) = lambda_initial
            ya[4] - rho_initial    # rho(ri) = rho_initial
        ])
    return boundary_conditions, a_initial_guess, nu_initial, llambda_initial, rho_initial
#
#
def tanh_transition_function(rho, rho_threshold=1, steepness=10):
    # Adjust the tanh function to smoothly transition from 0 to 1
    return 0.5 * (1 + np.tanh(steepness * (rho - rho_threshold)))
#
debug_info = []
#
#
def ode_system(r, y, P, dZetadXi):

    a, a_prime, nu, llambda, rho = y
    # switch the system when rho reaches 1
    # Debugging output
    if np.any(rho <= 1):
        debug_info.append((r, rho))
        return np.zeros_like(y)
    
    transition = tanh_transition_function(rho)

    #P_rho = np.where(rho>1, P(rho), 0)
    P_rho = P(rho)
    #dZetadXi_rho = np.where(rho>1, dZetadXi(rho),5)
    dZetadXi_rho = dZetadXi(rho)


    # Metric Potential equation
    expression_for_metric_pot = -1/r + np.exp(llambda)/r + (8 * np.exp(llambda) * G * np.pi * r * P_rho)/c**4 - \
                                (8 * np.exp(llambda) * fa * G * np.pi * r * (-fa * ma**2 * mu * (-1 + np.cos(a)) + \
                                c**2 * a * rho))/(c**4 * mu) + \
                                (4 * fa**2 * G * np.pi * r * a_prime**2)/c**4
    #
    dnu_dr = expression_for_metric_pot

    # Mass equation
    expression_for_mass = 1/r - np.exp(llambda)/r + (8 * np.exp(llambda) * G * np.pi * r * P_rho)/c**4 + \
                          (8 * np.exp(llambda) * fa * G * np.pi * r * (-fa * ma**2 * mu * (-1 + np.cos(a)) + \
                          c**2 * a * rho))/(c**4 * mu) + \
                          (4 * fa**2 * G * np.pi * r * a_prime**2)/c**4
    #
    dllambda_dr = expression_for_mass

    # Klein-Gordon equation (second-order turned first-order)
    expression_for_KG = ma**2 * np.exp(llambda) * np.sin(a) + (c**2 * rho)/(mu*fa) * np.exp(llambda) + \
                        a_prime * (-2/r + 1/2  * dllambda_dr - 1/2  * dnu_dr)
    #
    da_prime_dr = expression_for_KG

    # TOV equationß
    expression_for_TOV = transition * (-((fa * rho * ((3 * P_rho) / mu - (c**2 * rho) / mu) * a_prime) / (dZetadXi_rho * P_rho)) - \
                        (rho * (P_rho + c**2 * rho) * dnu_dr) / (2 * dZetadXi_rho * P_rho))
    #
    drho_dr = expression_for_TOV#np.where(rho>1, expression_for_TOV, 0)

    return np.array([a_prime, da_prime_dr, dnu_dr, dllambda_dr, drho_dr])
#
#
#
#
def central_densities(lower_limit, upper_limit, n_points):
    densities = np.logspace(np.log10(lower_limit), np.log10(upper_limit), n_points)
    return densities
#
#
#
