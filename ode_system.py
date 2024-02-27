import numpy as np
#
################# constants #################
gToGeV = 5.62e23 # (g/GeV)*)
cmInvToGeV = 1.98e-14 # (*cm^-1/GeV*) 
cmToGeVInv = 1/cmInvToGeV # (*GeV^-1/cm*)
gCm3ToGeV4 = (cmInvToGeV)**3 * gToGeV # (*g/cm^3 -> GeV^4*)
GeV4TogCm3 = 1/gCm3ToGeV4 # (*GeV^4 -> g/cm^3*)
sInvtoGeV = 6.58e-25 # (*GeV.s*)
GeVToErg = 1/(6.24e2) # (*Erg/GeV*)
PressueToGeV4 = gToGeV * cmInvToGeV * sInvtoGeV**2 # lol 
PhiFaGeVToCGs =(1/gToGeV /cmToGeVInv * (1/sInvtoGeV)**2)**(1/2) # (*(GeV . GeV^-1 . GeV^2)^(1/2) -> (g . cm . s^-2)*)
Msun = 1.988e33 # grams
G = 6.6743e-8 # (*dyne cm^2/g^2*)
c = 3e10 # (*cm/s*) 
fa = 1e15*PhiFaGeVToCGs # (*g^1/2 cm^1/2 s^-2*)
ma = 1e-11*1e-9 * cmToGeVInv; # (*1/L*)
NeutronMass = 0.939565 # (*GeV*) 
mu =(NeutronMass)/(3e-24) * PhiFaGeVToCGs # (*GeV*)
#
# 
################# initial guess around the minimum #################
def axion_initial_guess(rho_c):
    rho_c_over_rho_crit = (rho_c*c**2)/(fa*mu*ma**2)
    a_minimum = -np.arcsin(rho_c_over_rho_crit) 
    if rho_c_over_rho_crit>1: # for destabilization regime
        a_minimum = -1.
    print(f"rho_star/rho_crit = {rho_c_over_rho_crit:0.3e}")
    if rho_c_over_rho_crit<1:
        print(f"Minima exist.")
    else:
        print('Minima do not exist; entering the destabilization regime.')
    return a_minimum

################# calculate boundary conditions from expanding the equations to first order at small r #################
def create_boundary_conditions(eos_class, rho_c, nu_c, lambda_c, a_c, ri, verbose=0):
    """Create a boundary conditions function with specific initial conditions."""
    P = eos_class.get_pressure(extrapolate=True)
    dPdrho = eos_class.dP_dRho()
    nu_initial = nu_c + (4 * G * np.pi * ri**2 * \
                            (-2 * fa**2 * ma**2 * mu - 2 * a_c * c**2 * fa * rho_c + \
                            c**2 * mu * rho_c + 2 * fa**2 * ma**2 * mu * np.cos(a_c) + \
                                3 * mu * P(rho_c))) / (3 * c**4 * mu)
    
    llambda_initial = lambda_c + (8 * G * np.pi * ri**2 * \
                 (fa**2 * ma**2 * mu + a_c * c**2 * fa * rho_c + c**2 * mu * rho_c -  \
                  fa**2 * ma**2 * mu * np.cos(a_c))) / (3 * c**4 * mu)
    
    rho_initial = rho_c - 1/(6 * mu * dPdrho(rho_c)) * ri**2 * (
        (4 * G * np.pi * rho_c * \
         (-2 * fa**2 * ma**2 * mu - 2 * a_c * c**2 * fa * rho_c + c**2 * mu * rho_c + 2 * fa**2 * ma**2 * mu * np.cos(a_c) + 3 * mu * P(rho_c))) \
            / c**2 + (4 * G * np.pi * P(rho_c) * (-2 * fa**2 * ma**2 * mu - 2 * a_c * c**2 * fa * rho_c + \
                                           c**2 * mu * rho_c + 2 * fa**2 * ma**2 * mu * np.cos(a_c) + 3 * mu * P(rho_c))) / c**4 \
                                            + (c**2 * rho_c * (c**2 * rho_c + fa * ma**2 * mu * np.sin(a_c))) / mu - \
                                                  (3 * P(rho_c) * (c**2 * rho_c + fa * ma**2 * mu * np.sin(a_c))) / mu )
    a_initial_guess = a_c + (ri**2 * (c**2 * rho_c + fa * ma**2 * mu * np.sin(a_c))) / (6 * fa * mu)
    a_prime_initial = 0

    if verbose>0:
        print(f"\na_initial= {a_initial_guess:0.2e}" )
        print(f"a_prime_initial= {a_prime_initial:0.2e}" )
        print(f"nu_initial = {nu_initial:0.2e}")
        print(f"llambda_initial= {llambda_initial:0.2e}")
        print(f"rho_initial:{rho_initial:0.2e}\n")

    return  np.array([a_initial_guess, a_prime_initial, nu_initial, llambda_initial, rho_initial])
#
#
################# inside system with ivp and can find the radius in ~35s ################# 
def inside_ivp_system(r, y, P, dPdRho):
    #
    a, a_prime, nu, llambda, rho = y
    if rho < 0 or rho > 1e20:
        rho=0.
    #
    #
    # Metric Potential equation
    dnu_dr = -1/r + np.exp(llambda)/r + (8 * np.exp(llambda) * G * np.pi * r * P(rho))/c**4 - \
                                (8 * np.exp(llambda) * fa * G * np.pi * r * (-fa * ma**2 * mu * (-1 + np.cos(a)) + \
                                c**2 * a * rho))/(c**4 * mu) + \
                                (4 * fa**2 * G * np.pi * r * a_prime**2)/c**4
    #
    #
    # Mass equation
    dllambda_dr = 1/r - np.exp(llambda)/r + (8 * np.exp(llambda) * G * np.pi * r * rho)/c**2 + \
                          (8 * np.exp(llambda) * fa * G * np.pi * r * (-fa * ma**2 * mu * (-1 + np.cos(a)) + \
                          c**2 * a * rho))/(c**4 * mu) + \
                          (4 * fa**2 * G * np.pi * r * a_prime**2)/c**4
    #
    #
    # Klein-Gordon equation (second-order turned first-order)
    da_prime_dr = ma**2 * np.exp(llambda) * np.sin(a) + (c**2 * rho)/(mu*fa) * np.exp(llambda) + \
                        a_prime * (-2/r + 1/2  * dllambda_dr - 1/2  * dnu_dr)
    #
    #
    # TOV equation
    if rho>0:
        expression_for_TOV = - (P(rho) + c**2 * rho) * dnu_dr / 2 # GR part
        expression_for_TOV -=  fa / mu * a_prime * (3 * P(rho) - c**2 * rho) # Axion part: change this for different theories
        expression_for_TOV /= dPdRho(rho) # used chain rule to take P'(r) to rho'(r)
        
        drho_dr = expression_for_TOV
    else:
        drho_dr = 0

    return np.array([a_prime, da_prime_dr, dnu_dr, dllambda_dr, drho_dr])
#
#
################# stop when r steps have delta r<1e-8 ################# 
previous_r = [None]  # Use a list to allow modification inside the event function
def stop_at_small_r_step(r, y):
    # Access the global variable
    global previous_r
    # Initialize previous_r during the first call
    if previous_r[0] is None:
        previous_r[0] = r
        return 1  # Return a non-zero value to continue integration
    
    # Calculate the difference between the current and previous r values
    delta_r = np.abs(r - previous_r[0])
    
    # Update previous_r to the current r for the next call
    previous_r[0] = r
    
    # Check if the difference is less than the threshold
    if delta_r < 1e-8 :
        print("Event triggered: Stop at small r step")

        return 0  # Condition met, propose to stop integration
    else:
        return 1  # Condition not met, continue integration

# Make the event function terminal
stop_at_small_r_step.terminal = True
#
#
################# stop when rho<0 event ################# 
previous_rho = None  # Initialize previous_rho variable

def stop_at_rho_negative(r, y):
    global previous_rho  # Access the global variable

    # Extract the value of rho from y
    rho = y[-1]

    # Check if the difference is less than the threshold
    if rho < 0:
        print("Event triggered: Stop at negative rho")
        # Store the current value of rho
        previous_rho = rho
        return 0  # Condition met, propose to stop integration
    else:
        # Update the previous_rho variable
        previous_rho = rho
        return 1  # Condition not met, continue integration

# Make the event function terminal
stop_at_rho_negative.terminal = True
# will need later for MR relationship
def central_densities(lower_limit, upper_limit, n_points):
    densities = np.logspace(np.log10(lower_limit), np.log10(upper_limit), n_points)
    return densities
#
#
################# B.C. for outside system ################# 
def create_outside_bc(a_R, a_prime_R, nu_R, llambda_R):
    def outside_conditions(ya, yb):
        return np.array([
            ya[0] - a_R,                 #  ya[1] a_prime(ri) = 0
            yb[1],                 #  yb[1] a_prime(rf) = 0
            ya[2] - nu_R,    # nu(ri) = nu_initial
            ya[3] - llambda_R, # lambda(ri) = lambda_initial
        ])
    return outside_conditions
#
#
################# outside system ################# 
def outside_bvp_system(r, y):
    
    a, a_prime, nu, llambda = y

    
    # Metric Potential equation
    
    expression_for_metric_pot = -1/r + np.exp(llambda)/r  - \
                                (8 * np.exp(llambda) * fa * G * np.pi * r * (-fa * ma**2  * (-1 + np.cos(a))))/(c**4 ) + \
                                (4 * fa**2 * G * np.pi * r * a_prime**2)/c**4
    #
    dnu_dr = expression_for_metric_pot

    # Mass equation
    expression_for_mass = 1/r - np.exp(llambda)/r  + \
                          (8 * np.exp(llambda) * fa * G * np.pi * r * (-fa * ma**2  * (-1 + np.cos(a))))/(c**4) + \
                          (4 * fa**2 * G * np.pi * r * a_prime**2)/c**4
    #
    dllambda_dr = expression_for_mass

    # Klein-Gordon equation (second-order turned first-order)  
    expression_for_KG = ma**2 * np.exp(llambda) * np.sin(a) +  \
        a_prime * (-2/r + 1/2  * dllambda_dr - 1/2  * dnu_dr)
    #
    da_prime_dr = expression_for_KG



    return np.array([a_prime, da_prime_dr, dnu_dr, dllambda_dr])
#
# original full system. not working really but keeping just in case
def bvp_ode_system(r, y, P, dPdRho):

    a, a_prime, nu, llambda, rho = y
    # switch the system when rho reaches 1
    # Debugging output
    #if np.any(rho <= 1):
    #    debug_info.append((r, rho))
    #   return np.zeros_like(y)
    
    
    # Metric Potential equation
    expression_for_metric_pot = -1/r + np.exp(llambda)/r + (8 * np.exp(llambda) * G * np.pi * r * P(rho))/c**4 - \
                                (8 * np.exp(llambda) * fa * G * np.pi * r * (-fa * ma**2 * mu * (-1 + np.cos(a)) + \
                                c**2 * a * rho))/(c**4 * mu) + \
                                (4 * fa**2 * G * np.pi * r * a_prime**2)/c**4
    #
    dnu_dr = expression_for_metric_pot

    # Mass equation
    expression_for_mass = 1/r - np.exp(llambda)/r + (8 * np.exp(llambda) * G * np.pi * r * rho)/c**2 + \
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

    # TOV equation
    #expression_for_TOV =  -(fa / mu * a_prime * (3 * P(rho) - c**2 * rho ) + (P(rho) + c**2 * rho) * dnu_dr /2 ) / dPdRho(rho)
    expression_for_TOV = - (P(rho) + c**2 * rho) * dnu_dr / 2 # GR part
    expression_for_TOV -=  fa / mu * a_prime * (3 * P(rho) - c**2 * rho) # Axion part: change this for different theories
    expression_for_TOV /= dPdRho(rho) # used chain rule to take P'(r) to rho'(r)
    
    drho_dr = expression_for_TOV

    return np.array([a_prime, da_prime_dr, dnu_dr, dllambda_dr, drho_dr])
#
#