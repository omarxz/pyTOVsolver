import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Defining the class NeutronStarEOS
class NeutronStarEOS:
    def __init__(self, EoS): # to initialize the instance, we need EoS='APR' or 'SLY4'
        # Constants from the APR EOS
        self.extrapolate = False
        self.EoS = EoS.upper() # use the UPPERCASE of the input so it can be passed in any "format"
        self.default_xi = np.linspace(0, 50, 100000) # from rho=1 g/cm^3 to rho=1e16 g/cm^3, the full range of NS
        self.rhos = pow(10, self.default_xi)
        if self.EoS =='APR':
            self.set_APR_params()
        elif self.EoS=='SLY4':
            self.set_SLY4_params()
        else:
            raise ValueError("EoS must be either 'APR' or 'SLY4' ")
    # APR parameters   
    def set_APR_params(self):
        self.a1 = 6.22
        self.a2 = 6.121
        self.a3 = 0.006035
        self.a4 = 0.16354
        self.a5 = 4.73
        self.a6 = 11.5831
        self.a7 = 12.589
        self.a8 = 1.4365
        self.a9 = 4.75
        self.a10 = 11.5756
        self.a11 = -42.489
        self.a12 = 3.8175
        self.a13 = 2.3
        self.a14 = 14.81
        self.a15 = 29.80
        self.a16 =-2.976
        self.a17 = 1.99
        self.a18 = 14.93
    
    # SLy4 parameters
    def set_SLY4_params(self):
        self.a1  = 6.22
        self.a2  = 6.121
        self.a3  = 0.005925
        self.a4  = 0.16326
        self.a5  = 6.48
        self.a6  = 11.4971
        self.a7  = 19.105
        self.a8  = 0.8938
        self.a9  = 6.54
        self.a10 = 11.4950
        self.a11 = -22.775
        self.a12 = 1.5707
        self.a13 = 4.3
        self.a14 = 14.08
        self.a15 = 27.80
        self.a16 = -1.653
        self.a17 = 1.50
        self.a18 = 14.67

    # need this for EoS
    def f0(self, x):
        return 1 / (np.exp(x) + 1)

    # EoS parametrization
    def eq_of_state(self, xi=None, steepness=2): # not designed to be called
        term1 = ((self.a1 + self.a2*xi + self.a3*xi**3) / (1 + self.a4*xi)) * self.f0(self.a5 * (xi - self.a6))
        term2 = (self.a7 + self.a8*xi) * self.f0(self.a9 * (self.a10 - xi))
        term3 = (self.a11 + self.a12*xi) * self.f0(self.a13 * (self.a14 - xi))
        term4 = (self.a15 + self.a16*xi) * self.f0(self.a17 * (self.a18 - xi))
        return term1 + term2 + term3 + term4


    # returns interpolated pressure for solver. 
    # extrapolate=True let's you extrapolate to rho=0 but with negative pressures. 
    # maybe fix later if need to integrate to rho=0
    def get_pressure(self, extrapolate=None, steepness=14.322328):
        if extrapolate:
            self.extrapolate = extrapolate
            self.steepness = steepness
        if not isinstance(self.default_xi, Iterable) or isinstance(self.default_xi, str):
            raise ValueError("Input must be an iterable object like a list or a NumPy array.")
            
        # Calculate pressures using EoS
        log_pres = self.eq_of_state(self.default_xi)
        self.rhos = np.power(10, self.default_xi)
        self.pressures = np.power(10, log_pres)

        # if extrapolate=True, extrapolate the pressure to rho=0
        # not working well due to the introduction of singularities in dP_dRho later
        if extrapolate:
            # Fit a polynomial in the specified range (1 to 2)
            og_rhos = self.rhos
            og_pressures = self.pressures
            self.rho_tanh = np.linspace(0, 0.999, 500)
            #self.pressure_tanh = 2.18e6*np.tanh(self.rho_tanh*steepness)
            self.pressure_tanh = np.exp(self.rho_tanh*steepness) - 1
            # Combine extrapolated pressures with pressures from EoS
            combined_pressures = np.concatenate((self.pressure_tanh, self.pressures[self.rhos >= 1]))
            combined_rhos = np.concatenate((self.rho_tanh, self.rhos[self.rhos >= 1]))

            # Ensure unique and sorted values for combined_rhos
            self.combined_rhos, indices = np.unique(combined_rhos, return_index=True)
            unique_combined_pressures = combined_pressures[indices]
            # Create an interpolated function covering the entire range
            self.interp_pressure = interp1d(self.combined_rhos, unique_combined_pressures, kind='cubic')

            return self.interp_pressure
        else:
            self.interp_pressure = interp1d(self.rhos, self.pressures, kind='cubic')
            return self.interp_pressure

    # theoretical equation for dZetadXi from mathematica
    def dZetadXi(self):
        term1 = 1.4365 / (1 + np.exp(4.75 * (11.5756 - np.log10(self.rhos))))
        term2 = 3.8175 / (1 + np.exp(2.3 * (14.81 - np.log10(self.rhos))))
        term3 = -2.976 / (1 + np.exp(1.99 * (14.93 - np.log10(self.rhos))))
        term4 = (1.99 * np.exp(1.99 * (14.93 - np.log10(self.rhos))) * (29.8 - 2.976 * np.log10(self.rhos))) / (1 + np.exp(1.99 * (14.93 - np.log10(self.rhos))))**2
        term5 = (4.75 * np.exp(4.75 * (11.5756 - np.log10(self.rhos))) * (12.589 + 1.4365 * np.log10(self.rhos))) / (1 + np.exp(4.75 * (11.5756 - np.log10(self.rhos))))**2
        term6 = (2.3 * np.exp(2.3 * (14.81 - np.log10(self.rhos))) * (-42.489 + 3.8175 * np.log10(self.rhos))) / (1 + np.exp(2.3 * (14.81 - np.log10(self.rhos))))**2
        term7 = (6.121 + 0.018105 * np.log10(self.rhos)**2) / ((1 + np.exp(4.73 * (-11.5831 + np.log10(self.rhos)))) * (1 + 0.16354 * np.log10(self.rhos)))
        term8 = -0.16354 * (6.22 + 6.121 * np.log10(self.rhos) + 0.006035 * np.log10(self.rhos)**3) / ((1 + np.exp(4.73 * (-11.5831 + np.log10(self.rhos)))) * (1 + 0.16354 * np.log10(self.rhos))**2)
        term9 = -4.73 * np.exp(4.73 * (-11.5831 + np.log10(self.rhos))) * (6.22 + 6.121 * np.log10(self.rhos) + 0.006035 * np.log10(self.rhos)**3) / ((1 + np.exp(4.73 * (-11.5831 + np.log10(self.rhos))))**2 * (1 + 0.16354 * np.log10(self.rhos)))
        
        dZetdXi = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

        if self.extrapolate:
            #rho_to_0 = np.linspace(0,0.999,100)
            dZeta_to_0 = np.exp(-self.rho_tanh) + 2 * dZetdXi[0] - 5.47166064
            dZetdXi = np.concatenate([dZeta_to_0, dZetdXi])
            #self.rho_full = np.concatenate([rho_to_0, self.rhos])
            interp_dZeta_dXi = interp1d(self.combined_rhos, dZetdXi, kind='cubic')
            return interp_dZeta_dXi
        else:
            interp_dZeta_dXi_norm= interp1d(self.rhos, dZetdXi, kind='cubic')
            return interp_dZeta_dXi_norm

    """
    Code this here! Then compare to Mathematica and move forward if matches
    """
    # calculate dP/dRho that we need to change TOV from P'(r) to rho'(r)
    def dP_dRho(self):
        if not hasattr(self, 'interp_pressure'):
            raise ValueError("get_pressure must be called before dP_drho to initialize the interpolated pressure function.")
        dZeta = self.dZetadXi()
        P = self.interp_pressure
        if self.extrapolate:
            dPdRho = dZeta(self.combined_rhos[1:]) * P(self.combined_rhos[1:])/self.combined_rhos[1:]
            dPdRho = np.insert(dPdRho, 0, 82)
            self.interp_dPdRho = interp1d(self.combined_rhos, dPdRho, kind='cubic')
            return self.interp_dPdRho
        else:
            dPdRho = dZeta(self.rhos) * P(self.rhos)/self.rhos
            self.interp_dPdRho = interp1d(self.rhos, dPdRho, kind='cubic')
            return self.interp_dPdRho

    def plot_dZeta_dXi(self, log_scale=True, debug=False):
        interp_dZeta = self.dZetadXi()
        rhos = self.combined_rhos if self.extrapolate else self.rhos
        DZetaDXi = interp_dZeta(rhos)
        plt.figure(figsize=(10, 4))
        plt.plot(rhos, DZetaDXi, color='blue')
        plt.xlabel(r'$Log_{10}(\rho [g.cm^{-3}])$')
        plt.ylabel(r'$d\zeta/d\xi$')
        plt.title(self.EoS + ' Equation of State Gradient')
        if debug:
            plt.xlim(0, 1.5)
            plt.ylim(0,6)
            log_scale=False
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.grid(True)
        plt.show()

    # plot the EoS
    def plot_EoS(self, log_scale=True, debug=False):
        # Use the rhos and pressures from the class attributes
        interp_pres = self.interp_pressure
        rhos = self.combined_rhos if self.extrapolate else self.rhos
        pres = interp_pres(rhos)
        plt.figure(figsize=(10, 4))
        plt.plot(rhos, pres, color='blue')
        if debug:
            plt.xlim(0, 1.1)
            plt.ylim(0,1.75e6)
            log_scale = False
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.xlabel(r'$\rho (g.cm^{-3})$')
        plt.ylabel(r'$P (g.cm^{-1}.s^{-2})$')
        plt.title(self.EoS + ' Equation of State')
        plt.grid(True)
        plt.show()
    #
    # call to plot dP_drho
    def plot_dP_dRho(self, log_scale=True, debug=False):
        interp_dP_drho = self.dP_dRho()
        rhos = self.combined_rhos if self.extrapolate else self.rhos
        dP_drho = self.interp_dPdRho(rhos)
        plt.figure(figsize=(10, 4))
        plt.plot(rhos, dP_drho, color='blue')
        plt.xlabel(r'$\rho (g.cm^{-3})$')
        plt.ylabel(r'$dP/d\rho$')
        plt.title(self.EoS + ' Equation of State Gradient')
        if debug:
            plt.xlim(0.8, 3)
            plt.ylim(0,1e7)
            log_scale = False
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.grid(True)
        plt.show()

# old get_pressure() extrapolate
"""if extrapolate:
    # Fit a polynomial in the specified range (1 to 2)
    mask = (self.rhos >= 1) & (self.rhos <= 2) #2
    fit_rhos = np.insert(self.rhos[mask], 0, 0.0)
    fit_pressures = np.insert(self.pressures[mask], 0, 0.0)
    degree = 12 #12
    coeffs = np.polyfit(fit_rhos, fit_pressures, degree)
    polynomial = np.poly1d(coeffs)

    # Extrapolate pressure for rho < 1 using the polynomial
    rho_extrapolate = np.linspace(0, max(fit_rhos), 100000)
    extrapolated_pressure = polynomial(rho_extrapolate)

    # Combine extrapolated pressures with pressures from EoS
    combined_pressures = np.concatenate((extrapolated_pressure, self.pressures[self.rhos >= 1]))
    combined_rhos = np.concatenate((rho_extrapolate, self.rhos[self.rhos >= 1]))

    
    # Ensure unique and sorted values for combined_rhos
    self.combined_rhos, indices = np.unique(combined_rhos, return_index=True)
    unique_combined_pressures = combined_pressures[indices]
    print(f"At rho={self.combined_rhos[0]}, P = {unique_combined_pressures[0]}")
    # Create an interpolated function covering the entire range
    self.interp_pressure = interp1d(self.combined_rhos, unique_combined_pressures, kind='cubic')"""