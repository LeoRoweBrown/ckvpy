import numpy as np
from scipy.integrate import simps
import scipy.constants as const

def compute(theta, f, beta, L, n=None):
    """compute number of photons due to Frank-Tamm and Fresen equations
    theta (ndarray/list[float]): Angles in chosen wavelength range
    f (ndarray/list[float]): Frequencies in chosen wavelength range
    n (ndarray/list[float]): Refractive index in chosen wavelength range
    beta (float): Ratio of electron speed to speed of light
    """
    if n is None:
        print("Using Cherenkov angle to derive n instead of d(omega)/dk")
        n = 1/(beta*np.cos(theta))
    r_s = np.absolute(
        (n*np.cos(theta) - np.sqrt(1-(n*np.sin(theta)**2.)))/ \
        (n*np.cos(theta) + np.sqrt(1-(n*np.sin(theta)**2.)))
        )
    r_p = np.absolute(
        (n*np.sqrt(1-(n*np.sin(theta)**2.)) - np.cos(theta))/ \
        (n*np.sqrt(1-(n*np.sin(theta)**2.)) + np.cos(theta))
        )
    r_eff =(r_p + r_s)/2.
    print(r_eff)
    t_eff = 1-r_eff
    print("Transmission coeff:", t_eff)
    # derive angles inside medium with snell's law for Fresnel equation
    theta_in = np.arcsin(n*np.sin(theta))
    # n_photons = \
    #     (const*fine_structure/(const.hbar*const.c**2.))*\
    #     simps((1-1./(beta**2.*n**2.))*t_eff, x=const.h*f)
    # need even spaced intervals -> interpolate
    f_interp = np.linspace(np.min(f), np.max(f), num=30)
    theta_interp = np.interp(f_interp, f, theta_in)
    t_eff_interp = np.interp(f_interp, f, t_eff)
    n_photons = \
        L*(const.fine_structure/(const.hbar*const.c))* \
        simps(np.sin(theta_interp)**2.*t_eff_interp*const.h, x=f_interp)
    print(n_photons, "photons")
    return n_photons