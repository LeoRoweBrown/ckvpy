import numpy as np
import scipy.constants as const
import json
import os
from matplotlib import pyplot as plt
import ckvpy.tools.photon_yield as photon_yield
import ckvpy.tools.effective as effective
from ckvpy.tools.analysis import dataAnalysis

class dataAnalysis3D(dataAnalysis):
    """Data analysis class for 3D models, requires intersection of electron
    plane and dispersion to find cherenkov angle etc."""
    def __init__(self, data):
        self.data_full = data  # includes full Brillouin zone data
        self.data_dict = {}  # same structure as in 2D case
        self._init_data_dict(data)
        self._get_num_bands()
        self.status = {
            'reflected': True,
            'interpolated': True,
            'intersected': False
        }
    
    def _init_data_dict(self, data):
        self.data_dict = {}
        for root in self.data_full:
            self.data_dict[root] = {}
            for band in self.data_full[root]:
                self.data_dict[root][band] = {}

    def calculateCherenkov(self, beta=0.999, direction = [1,0]):
        """Find intersection of electron plane and dispersion to get
        Cherenkov behaviour
        Args:
            beta (float): electron speed ratio with c
            direction (list): determines direction of electron with idices
                rho (|x,y|) and z which defines e-plane omega = k.v
        """
        if type(direction[0]) is not int or type(direction[1]) is not int:
            raise ValueError("Only directions purely in z or rho supported")
        for band in self.data_full['default']:
            m_rho = self.data_full['default'][band]['mi']  # matrix of k_rho values
            mz = np.copy(self.data_full['default'][band]['mz'])  # mutated so copy
            my = self.data_full['default'][band]['my']  # matrix of ky values
            mx = self.data_full['default'][band]['mx']  # matrix of kx values
            mf = np.copy(self.data_full['default'][band]['mf'])  # mutated so copy

            z_array = mz.T[0][-1:1:-1]  # starts at maximum
            rho_array = m_rho[0][1:-1]  # cut off edges (interp)

            # e_plane = self.data_dict['default'][band]['mj']*3.e8*v
            mf *= 2*np.pi  # omega=2pif
            mf = mf.T # since we transpose z to get z array from columns
            self.data_dict['default'][band] = \
                {'kz': [None], 'k_rho': [None], 'frequency': [None], 'direction': direction}

            kz_c = np.array([])  # empty temp arrays to store crossing points
            k_rho_c = np.array([])
            f_c = np.array([])
            for kz_i, kz in enumerate(z_array[:-1]):  # ith value of kz
                for k_rho_i, k_rho in enumerate(rho_array[:-1]):  # jth k_rho
                    kz2 = z_array[kz_i + 1]  # i+1th value of kz
                    k_rho2 = rho_array[k_rho_i + 1]  # j+1th k_rho
                    f = mf[kz_i, k_rho_i]  # f(kz,k_rho)
                    fz2 = mf[kz_i + 1, k_rho_i]  # f(kz2,k_rho)
                    f_rho2 = mf[kz_i, k_rho_i + 1]  # f(kz,k_rho2)
                    # get crossing points and booleans (was crossing found?)
                    rho_found, rho_cross, z_found, z_cross = \
                        self._cross(beta, kz, kz2, k_rho, k_rho2, f, fz2,
                                    f_rho2, direction)
                    k_rho_cross, f_rho_cross = rho_cross
                    kz_cross, fz_cross = z_cross
                    if z_found:  # crossing found in kz direction
                        kz_c = np.append(kz_c, kz_cross)
                        k_rho_c = np.append(k_rho_c, k_rho)
                        f_c = np.append(f_c, fz_cross)
                    if rho_found:  # crossing found in k_rho direction
                        kz_c = np.append(kz_c, kz)
                        k_rho_c = np.append(k_rho_c, k_rho_cross)
                        f_c = np.append(f_c, f_rho_cross)
            self.data_dict['default'][band]['kz'] = kz_c.tolist()
            self.data_dict['default'][band]['k_rho'] = k_rho_c.tolist()
            # set back to f instead of omega
            self.data_dict['default'][band]['frequency'] = \
                (f_c/(2*np.pi)).tolist()
            if len(self.data_dict['default'][band]['kz']) == 0:
                raise Warning("No intersection found between electron plane "
                            "and dispersion plane,")
        self.status['intersected'] = True
        # self._rm_nan()  # remove NaNs # needs fixing for 3D case (add bands key)
        self._comp_angle()

    def _cross(self, beta, kz, kz2, k_rho, k_rho2, f, fz2,
               f_rho2, direction):
        """Find crossings between electron plane in omega=v_z*kz+v_rho*k_rho
        and dispersion to find Cherenkov modes Interpolates between kz, kz2
        and k_rho, k_rho2 separately to find them.
        
        Args: 
            beta (float): speed as percentage of c
            direction (list): from 0 to 1, component in rho and z direciton 
                respectively in form [float, float] ([1,1] -> velocity=(v,v))
            kz, kz2, k_rho, k_rho2 (float): ith and i+1th value of kz/k_rho
                in ascending order
            f (float): value of frequency at kz k_rho
            fz2, f_rho2 (float): value of frequency at (k_rho,kz2) and 
                (k_rho2,kz)

        Returns:
            z_found, rho_found (bool): True if crossing found looking in the 
                kz/k_rho direction
            (kz_cross, fz_cross) tuple(float): kz and fz where crossing found
            (k_rho_cross, f_rho_cross) tuple(float) same for k_rho and f_rho
        """

        # first look along kz:
        ve = beta*const.c  # speed of light
        m_z = (fz2-f)/(kz2-kz)  # gradient in kz direction
        m_rho = (f_rho2-f)/(k_rho2-k_rho)  # gradient in k_rho direction
        # electron speed components
        v_rho, v_z = ve*direction[0], ve*direction[1]  
        v_abs = (v_rho**2 + v_z**2)**0.5

        cz = f - m_z*kz  # f intercept constant in f=m*k+c
        c_rho = f - m_rho*k_rho
        # first look at kz direction
        if abs(m_z - v_z) < 1e-15*abs(v_rho*k_rho - cz):
            z_found = False  # m -> +-infinity
            kz_cross = fz_cross = None
        else:
            kz_cross = (v_rho*k_rho - cz)/(m_z - v_z)
            fz_cross = kz_cross*m_z + cz
            z_found = True
        z = (kz_cross, fz_cross)

        if abs(m_rho - v_rho) < 1e-20*abs(v_z*kz - c_rho):
            rho_found = False  # m -> +-infinity
            k_rho_cross, f_rho_cross = None, None
        else:
            k_rho_cross = (v_z*kz - c_rho)/(m_rho - v_rho)
            f_rho_cross = k_rho_cross*m_rho + c_rho
            rho_found = True
        rho = (k_rho_cross, f_rho_cross)

        if rho_found:  # check if in range that interpolation is valid
            k_bounds = k_rho_cross >= min(k_rho,k_rho2) and \
                k_rho_cross <= max(k_rho,k_rho2)
            f_bounds = f_rho_cross >= min(f,f_rho2) and \
                f_rho_cross <= max(f,f_rho2) and f_rho_cross > 0.

            if k_bounds and f_bounds : 
                rho_found = True
            else:
                rho_found = False

        if z_found:
            k_bounds = kz_cross >= min(kz,kz2) and \
                kz_cross <= max(kz,kz2)
            f_bounds = fz_cross >= min(f,fz2) and \
                fz_cross <= max(f,fz2) and fz_cross > 0.
            if k_bounds and f_bounds:
                z_found = True
            else:
                z_found = False
        
        return rho_found, rho, z_found, z

    def _comp_angle(self):
        """Find angles *outside* crystal using k-vectors."""
        # everything else hard-codes 'default', might change
        for root in self.data_dict:
            for band in self.data_dict[root]:
                kz = np.array(self.data_dict[root][band]['kz'])
                k_rho = np.array(self.data_dict[root][band]['k_rho'])
                f = np.array(self.data_dict[root][band]['frequency'])
                d_rho, dz = self.data_dict[root][band]['direction']
                # adj_for_e_diretion = np.arctan(dz/(d_rho+1e-20))
                # theta = np.arctan(kz/(k_rho+1e-20)) - adj_for_e_diretion
                k0 = np.sqrt(kz*kz + k_rho*k_rho)
                # dz = 1, k_rho cons
                if dz == 1: k_parallel = k_rho
                elif d_rho == 1: k_parallel = kz
                # print(k_parallel)
                # print(k_rho)
                theta = np.arcsin(k_parallel/k0)
                 #print(theta)
                wl = const.c/np.array(f)
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(k_rho, kz, f, color='black')
                # plt.show()

                self.data_dict[root][band]['wavelength'] = wl.tolist()
                self.data_dict[root][band]['angle'] = theta.tolist()
                self.wl_cut(root, band, wl_range=[0.,1000e-9],\
                    sign=1, param_key='all', mutate=True)
                self.calculate_n_eff()
                # print(print(wl)
                # print(f)
                # wl_interp1, wl_interp2, mean, err = \
                #     self.calc_err(wl_range)
        
