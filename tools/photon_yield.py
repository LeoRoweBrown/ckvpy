import numpy as np
from scipy.integrate import simps
import scipy.constants as const
import os

def compute(theta_in, f, beta, L, n=None, qe="hpd", mirror1=0.9, mirror2=0.9):
    """compute number of photons due to Frank-Tamm and Fresen equations
    theta (ndarray/list[float]): Angles in chosen wavelength range
    f (ndarray/list[float]): Frequencies in chosen wavelength range
    n (ndarray/list[float]): Refractive index in chosen wavelength range
    beta (float): Ratio of electron speed to speed of light

    TODO: replace n = 1/(beta*np.cos(theta_in)) with actual n_eff
    """
    if n is None:
        print("Using Cherenkov angle to derive n instead of d(omega)/dk")
        n = 1/(beta*np.cos(theta_in))
    r_s = np.absolute(
        (n*np.cos(theta_in) - np.sqrt(1-(n*np.sin(theta_in)**2.)))/ \
        (n*np.cos(theta_in) + np.sqrt(1-(n*np.sin(theta_in)**2.)))
        )
    r_p = np.absolute(
        (n*np.sqrt(1-(n*np.sin(theta_in)**2.)) - np.cos(theta_in))/ \
        (n*np.sqrt(1-(n*np.sin(theta_in)**2.)) + np.cos(theta_in))
        )
    r_eff =(r_p + r_s)/2.
    # print(r_eff)
    t_eff = 1-r_eff
    print("Transmission coeff:", t_eff)
    # derive angles inside medium with snell's law for Fresnel equation
    # theta_in = np.arcsin(n*np.sin(theta))
    # n_photons = \
    #     (const*fine_structure/(const.hbar*const.c**2.))*\
    #     simps((1-1./(beta**2.*n**2.))*t_eff, x=const.h*f)
    # need even spaced intervals -> interpolate
    # integral is over f
    f_interp = np.linspace(np.min(f), np.max(f), num=30)
    theta_interp = np.interp(f_interp, f, theta_in)
    t_eff_interp = np.interp(f_interp, f, t_eff)
    qe_file = os.path.join(os.path.dirname(__file__),\
        "..\\qe\\"+ qe + ".txt")
    print("Reading from", qe_file)
    f_qe, qe = np.loadtxt(qe_file).T
    qe_interp = np.interp(f_interp, f_qe, qe/100.)
    print("Quantum efficiency", qe_interp)
    n_photons = \
        L*(const.fine_structure/(const.hbar*const.c))* \
        simps(np.sin(theta_interp)**2.*t_eff_interp*qe_interp
        *mirror1*mirror2*const.h, x=f_interp)
    print(n_photons, "photons")
    return n_photons