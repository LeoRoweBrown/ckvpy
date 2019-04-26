import numpy as np
from matplotlib import pyplot as plt
import os

# TODO: change to deal purely with refractive index (not dielectric constant)
# no lomnger using Maxwell-Garnett?
def maxwell_garnett_index(E_matrix, E_inclusion, ratio):
    """
    variables:
    E_matrix: (float) dielectric constant of matrix material
    E_inclusion: (float) dielectric constant of inclusion material
    ratio: (float) volume ratio between inclusion and overall volume (matrix)
    returns (float) effective dielectric constant
    """
    if type(E_matrix) is np.ndarray:
        E_inclusion = np.ones_like(E_matrix)*E_inclusion
    elif type(E_inclusion) is np.ndarray:
        E_matrix = np.ones_like(E_inclusion)*E_matrix
    elif type(E_inclusion) is not float and type(E_matrix) is not float:
        try:
            E_inclusion = float(E_inclusion)
            E_matrix = float(E_matrix)
            ratio = float(ratio)
        except:
            raise TypeError("Values can be floats or numpy.ndarray only")
    E_effective = E_matrix*(2*ratio*(E_inclusion-E_matrix) + E_inclusion +\
        2*E_matrix)/(2*E_matrix + E_inclusion - ratio*(E_inclusion-E_matrix))
    return E_effective

def ratio(E_matrix, E_inclusion, E_effective):
    if type(E_inclusion) is not float or type(E_matrix) is not float \
        or type(E_effective) is not float:
        try:
            E_inclusion = float(E_inclusion)
            E_matrix = float(E_matrix)
            E_effective = float(E_effective)
        except:
            raise TypeError("Values can be floats only")

    return (E_effective*(2*E_matrix+E_inclusion) - \
        E_matrix*(E_inclusion+2*E_matrix))/(E_effective*(E_inclusion-E_matrix)\
        + 2*E_matrix*(E_inclusion-E_matrix))

def average_index(n1, n2, v1=None, v2=None, vr=None):
    """Compute volume weighted average refractive index
    Args:
        n1, n2 (float/ndarray): refractive index of material 1 and 2
        v1, v2 (float): volume of materials 1 and 2
        vr (float): volume ratio of v1/(v1+v2) used instead of v1 and v2
    """ 
    if vr is None:
        if v1 or v2 is None:
            raise ValueError("Please supply volumes v1, v2 "
                             "or volume ratio vr")
        n = (v1*n1 + v2*n2)/(v1+v2)
    else:
        # n = (vr*n1 + n2)/(vr+1) when vr = v1/v2
        n = vr*n1 + (1-vr)*n2
    return n

def compare_medium(n_data, th_in, wl_in, ratio, index="sio2", \
    band=0, filename=None, modelname=None, \
    n_lim=None, beta=0.999):
    """Compare expected refractive index/Cherenkov angle from 
    Maxwell-Garnett formula to data from simulation. Analysis is valid
    INSIDE the crystal, so wavelength derived from k not c/f
    Args:
        n_data: refractive index calculated by sqrt(kx*kx+ky*ky+kz*kz)/k0
        wl: wavelength inside the crystal, i.e. 2pi/k not c/f
        index (str): material refractive index in ./index/<index>.txt
            used to calculate expected index from effective medium.
        ratio_2d (float): ratio of dielectric to air in the plane
        ratio_3d (float): ratio of dielectric to air out of plane
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['font.size'] = 14
    index_file = os.path.join(os.path.dirname(__file__),\
        "..\\index\\"+ index + ".txt")
    print("Reading from", index_file)

    wl_sio2, n_sio2 = np.loadtxt(index_file).T
    # print(wl) # TODO: Handle NaNs in data
    ind = np.argsort(wl_in)
    wl_in = np.array([wl_in[i] for i in ind])
    th_in = np.array([th_in[i] for i in ind]) # unused at the moment
    n_sio2_interp = np.interp(wl_in, wl_sio2, n_sio2)
    e_sio2 = n_sio2_interp*n_sio2_interp
    # volume ratio 2d
    # volume ratio z direction
    # eff_2d = e_index(1.0, e_sio2, ratio_2d)  # 2d
    # n_test = eff_test**0.5
    # if ratio_3d is None:
    #     eff = eff_2d
    # else:
    #     eff = e_index(1.0, eff_2d, ratio_3d)  # full 3d
    # n_eff = eff**0.5  # n = sqrt(eps)
    # n_data = 1./(np.cos(th)*beta)  # Cherenkov formula
    # n_test = average_index(n_sio2_interp, 1.0, vr=0.106)
    n_mg = np.sqrt(maxwell_garnett_index(1.0, e_sio2, ratio))
    n_eff = average_index(n_sio2_interp, 1.0, vr=ratio)
    # print(wl_in)
    # th_eff = np.arccos(1./beta*n_eff)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot(wl_in*1e9, n_eff, label="Effective medium theory "
        "(Volume-weighted Average)", \
        color='black', linestyle='dotted', marker='None', markersize=6)
    ax.plot(wl_in*1e9, n_data,
        label="Simulation",  linestyle='None', color='black', marker='o',\
        markersize=6)
    ax.plot(wl_in*1e9, n_mg, label="Effective medium theory "
        "(Maxwell-Garnett)",
        color='black', linestyle='--', marker='None', markersize=6)
    # ax.set_xlim([np.min(wl_in),1000])
    ax.set_xticks(np.arange(np.min(wl_in)-100, 1000+100, 100))
    # ax.set_ylim([np.min(n_eff)-0.005, np.max(n_data)+0.005])
    global_max = max([np.max(n_eff), np.max(n_data)])
    global_min = min([np.min(n_eff), np.min(n_data)])

    # ax.set_ylim([np.min(n_eff)-0.005, 1.07])
    if n_lim is None:
        n_lim = global_min, global_max
    ax.set_ylim(n_lim)
    ax.set_xlim([0,1000])

    title = ("Effective Index Comparison Between Theory and "
            "Simulation for \n (Band " + str(int(band)+1) + ")")
    if modelname is not None:
        title += " (" + modelname + ")"
    ax.set_title(title)
    ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax.set_ylabel(r"Refractive index $n_{eff}$")
    ax.legend()
    # ax1 = ax.twinx() # fig.add_subplot(212)
    # ax1.set_xlim([np.min(wl_in),600])
    
    yl = np.arccos(1./(beta*n_lim[0]))
    yh = np.arccos(1./(beta*n_lim[1]))
    # ax1.set_ylim([yl, yh])
    # ax1.set_yticks(np.arange(np.round(yl*2.0, 2)/2.0, \
    #     yh+0.005, 0.01))
    # ax1.set_xlabel(r"Wavelength $\lambda$ (nm)")
    # ax1.set_ylabel(r"Saturated Cherenkov Angle $\theta_c$ (rad)")
    if filename is None:
        fig.savefig("untitled_effective_index"+str(band)+"test.png")
    else:
        fig.savefig(filename+str(band)+".png")
    plt.close()
    return n_mg, n_eff