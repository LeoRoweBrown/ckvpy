import numpy as np
import scipy.integrate
import scipy.constants as const

def compute_yield(theta, f, n, beta):
    r_s = np.abs(
        (n*np.cos(theta) - np.sqrt(1-(n*np.sin(theta)**2.)))/ \
        (n*np.cos(theta) + np.sqrt(1-(n*sin(theta)**2.)))
        )
    r_p = np.abs(
        (n*np.sqrt(1-(n*np.sin(theta)**2.)) - np.cos(theta))/ \
        (n*np.sqrt(1-(n*np.sin(theta)**2.)) + np.cos(theta))
        )
    r_eff = (r_p + r_s)/2.
    t_eff = 1-r_eff
    p_yield = (1./const.hbar)