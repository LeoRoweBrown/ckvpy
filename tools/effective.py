import numpy as np
from matplotlib import pyplot as plt
import os

def e_index(E_matrix, E_inclusion, ratio):
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
    elif type(E_inclusion) is float and type(E_matrix) is float:
        pass
    else:
        raise TypeError("Values can be floats and numpy arrays only")

    return E_matrix*(2*ratio*(E_inclusion-E_matrix) + E_inclusion +\
        2*E_matrix)/(2*E_matrix + E_inclusion - ratio*(E_inclusion-E_matrix))

def ratio(E_matrix, E_inclusion, E_effective):
    if type(E_inclusion) is not float or type(E_matrix) is not float \
        or type(E_effective) is not float:
        raise TypeError("Values can be floats only")

    return (E_effective*(2*E_matrix+E_inclusion) - \
        E_matrix*(E_inclusion+2*E_matrix))/(E_effective*(E_inclusion-E_matrix)\
        + 2*E_matrix*(E_inclusion-E_matrix))

def compare_medium(th, wl, ratio_2d, ratio_3d=None, index="sio2", \
    wl_range = [250.e-9, 500.e-9], band=0, filename=None, modelname=None):
    """Compare expected refractive index/Cherenkov angle from 
    Maxwell-Garnett formula to data from simulation.
    Args:
        index (str): material refractive index in ./index/<index>.txt
            used to calculate expected index from effective medium.
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['font.size'] = 14
    index_file = os.path.join(os.path.dirname(__file__),\
        "..\\index\\"+ index + ".txt")
    print("Reading from", index_file)

    wl_sio2, n_sio2 = np.loadtxt(index_file).T
    # print(wl)
    ind = np.argsort(wl)
    wl = np.array([wl[i] for i in ind])
    th = np.array([th[i] for i in ind])
    n_sio2_interp = np.interp(wl, wl_sio2, n_sio2)
    e_sio2 = n_sio2_interp*n_sio2_interp
    # volume ratio 2d
    v_r = (np.pi*(0.45*(3**0.5))**2)/(3*3**0.5/2)
    # volume ratio z direction
    v_rz = 2./3.
    eff_2d = e_index(e_sio2, 1.0, ratio_2d)  # 2d slab
    if ratio_3d is None:
        eff = eff_2d
    else:
        eff = e_index(eff_2d, 1.0, ratio_3d)  # full 3d
    n_eff = eff**0.5  # n = sqrt(eps)
    n_data = 1./np.cos(th)  # Cherenkov formula
    th_eff = np.arccos(1./n_eff)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot(wl*1e9, n_eff, label="Effective medium theory", \
        color='black', linestyle='dotted', marker='*', markersize=6)
    ax.plot(wl*1e9, n_data,
        label="Simulation", color='black', marker='o',\
        markersize=6)
    ax.set_xlim([np.min(wl),1000])
    ax.set_xticks(np.arange(np.min(wl)-100, 1000+100, 100))
    # ax.set_ylim([np.min(n_eff)-0.005, np.max(n_data)+0.005])
    global_max = max([np.max(n_eff), np.max(n_data)])
    global_min = min([np.min(n_eff), np.min(n_data)])

    ax.set_ylim([np.min(n_eff)-0.005, 1.07])
    ax.set_ylim(global_min, global_max)

    title = ("Effective Index Comparison Between Theory and "
            "Simulation \n (Band " + str(int(band+1)) + ") ")
    if modelname is not None:
        title += "(" + modelname + ")"
    ax.set_title(title)
    ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax.set_ylabel(r"Refractive index $n_{eff}$")
    ax.legend()
    ax.axvline(wl_range[0], linestyle='dashed', color='black')
    ax.axvline(wl_range[1], linestyle='dashed', color='black')
    ax1 = ax.twinx() # fig.add_subplot(212)
    ax1.set_xlim([np.min(wl),1000])
    # ax1.yaxis.grid(color='black')
    # ax.xaxis.grid(color='black')
    # ax1.set_ylim([np.arccos(1./(np.min(n_eff)-0.005)), np.arccos(1./(np.max(n_data)+0.005))])
    yl = np.arccos(1./(np.min(n_eff)-0.005))
    yh = np.arccos(1./np.max(n_eff))
    ax1.set_ylim([yl, yh])
    ax1.set_yticks(np.arange(np.round(yl*2.0, 2)/2.0, \
        yh+0.005, 0.01))
    ax1.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax1.set_ylabel(r"Saturated Cherenkov Angle $\theta_c$ (rad)")
    if filename is None:
        fig.savefig("effective_index"+str(band)+".png")
    else:
        fig.savefig(filename+str(band)+".png")
    # fig.show()