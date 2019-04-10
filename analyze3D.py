# TODO: Add filename options for plots
# TODO: Maybe add option for a non-default sort_key (e.g. for different a_z)

import numpy as np 
import csv
import os
import math
from scipy import interpolate
import scipy.constants as const
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import ckvpy.tools.effective as effective
import ckvpy.tools.photon_yield as photon_yield
from ckvpy.tools.bz2d import Bzone2D

__all__ = ['Analyze3D']

class Analyze3D():
    """Intersect 2D band structure with electron plane. Can be used for 3D 
    data or 2D data, but 3D data must be confined to a 2D plane which at
    the moment is constrained to just planes in k_rho,kz where 
    k_rho = norm(kx,ky). Realistically in COMSOl this means only kz,kx and
    kz,ky because of the way bands are found: a direction of k is chosen
    and increase |k|. Always sorted by band (band is the root key)""" 
    def __init__(self, datafile, symmetry=4, headers=['skip', 'band', 'skip',
                 'frequency', 'kx', 'ky', 'kz', 'n', 'skip'],
                 ndim=3, resolution=100):

        # plt.rcParams.update({'font.size': 14})
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.axisbelow'] = True
        np.set_printoptions(precision=3) # will this affect data saved to text?
        self.ndim = ndim
        for header in headers:
            if header not in \
            ['band', 'frequency', 'kx', 'ky', 'kz', 'n', 'skip']:
                print(header, "not allowed.")
                raise ValueError(
                    "Invalid header supplied, must be one of "
                    "['band', 'frequency', 'kx', 'ky', 'kz', 'n']")
        
        data_loader = Bzone2D(datafile, headers=headers, ndim=ndim,
                              symmetry=symmetry, resolution=resolution)
        self._init_analysis(data_loader)


    def _init_analysis(self, data_loader):
        """Pass data dictionary to dataAnalysis object"""
        self.data = dataAnalysis(data_loader.data) 
        # TODO: now replace every self.calc_err 
        # with data.calc_err etc.

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

    def analyze_error(self, band, wl_range=[250.e-9, 500.e-9], 
                     theory_compare=True, wavelength_range=True):
        """Calculate average cherenkov angle and error from angle against
        wavelength data. Then have option to compare to effective medium
        theory and calculate for different wavelength ranges."""

        kz = self.data['default'][band]['cherenkov']['kz']
        k_rho = self.data['default'][band]['cherenkov']['k_rho']
        f = self.data['default'][band]['cherenkov']['frequency']
        d_rho, dz = self.data['default'][band]['cherenkov']['direction']
        adj_for_e_diretion = np.arctan(dz/(d_rho+1e-20))
        theta = np.arctan(kz/(k_rho+1e-20)) - adj_for_e_diretion
        # then compute outside angle
        np.tan(theta)
        # wl = 2*np.pi*3.e8/f
        # wl = 2.*np.pi/(kz**2.+k_rho**2.+1e-7)**0.5
        wl = const.c/f
        # print(print(wl)
        # print(f)
        pos_th, pos_wl, mean, err = \
            self._calc_err(theta, wl, wl_range)
        neg_th, neg_wl, neg_mean, neg_err = \
            self._calc_err(theta, wl, wl_range, sign=-1)
        self.data['default'][band]['cherenkov']['angle'] = theta
        self.data['default'][band]['cherenkov']['pos'] = {'angle': pos_th}
        self.data['default'][band]['cherenkov']['pos']['wavelength'] = pos_wl
        self.data['default'][band]['cherenkov']['pos']['error'] = err
        self.data['default'][band]['cherenkov']['pos']['average'] = mean
        self.data['default'][band]['cherenkov']['neg'] = {'angle': neg_th}
        self.data['default'][band]['cherenkov']['neg']['wavelength'] = neg_wl
        self.data['default'][band]['cherenkov']['neg']['error'] = neg_err
        self.data['default'][band]['cherenkov']['neg']['average'] = neg_mean
        self.data['default'][band]['cherenkov']['wavelength'] = wl

    def plotRange(self):
        for band in self.data['default']:
            wl_low_a = []
            mean_a = []
            err_a = []
            theta = self.data['default'][band]['cherenkov']['angle']
            # self.sort_data('wavelength', subkeys=['cherenkov'])
            wl = np.array(self.data['default'][band]['cherenkov']['wavelength'])
            # print(wl)
            # print(theta)
            for i, wl_low in enumerate(range(250,401,50)):
                print(wl_low)
                wl_r = [wl_low*1.e-9, 500.e-9]
                _, _, mean_t, err_t = \
                    self._calc_err(theta, wl, wl_r)
                if err_t is None or mean_t is math.nan:
                    continue
                mean_a.append(mean_t)
                err_a.append(err_t)
                wl_low_a.append(wl_low)
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            print(mean_a)
            print(err_a)
            mean_a = np.array(mean_a)
            err_a = np.array(err_a)
            plot1 = ax.errorbar(wl_low_a, mean_a, err_a/2., \
                color='black', \
                capsize=5, marker='o', markersize=5, label='Cherenkov Angle')
            ax.set_title("Effect of Wavelength Range on Cherenkov Angle and "
                         "\n Chromatic Error (3D Model)")
            ax.set_xlabel(r"Lower Wavelength Limit (nm)")
            ax.set_ylabel(r"Average Saturated Cherenkov Angle $\theta_c$ "
                          r"(rad)")
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.001))
            ax1 = ax.twinx()
            plot2 = ax1.plot(wl_low_a, err_a, color='black', \
                linestyle='--', label='Chromatic Error')
            ax1.set_ylabel(r"Chromatic Error $\Delta\theta_c$ (rad)")
            #ax.set_yticks(np.arange(np.min(mean)-np.max(err), \
            #    np.max(mean)+np.max(err), 0.001))
            fig.legend(loc=1, bbox_to_anchor=(1,1), \
                bbox_transform=ax.transAxes)
            fig.savefig("wavelengths.png", bbox_inches='tight')
            fig.show()
            plt.close()

    def _calc_err(self, theta, wl, wl_range, sign=1):
        """Find chromatic error and remove negative/positive angles between
        for wavelengths between 250nm and 500nm.
        Args:
            theta (array[float]): array of cherenkov angles found from 
                intersection
            wavelength (array[float]) array of wavelengths assoicated
                with theta array
            sign (int): whether to accept positive (1) or negative (-1) angles

        Returns:
            (list), (list), (float), (float): list of angles between 
            250-500nm, list of wavelengths for these angles, mean of
            angles and range of angles
        """
        wl_nm_range = []
        theta_nm_range = []
        mean = None
        err = None
        for i, w in enumerate(wl):
            if w < wl_range[1] and w > wl_range[0] and sign*theta[i]>0:
                wl_nm_range.append(w)
                theta_nm_range.append(theta[i])
        try:
            mean = np.mean(theta_nm_range)
            err = abs(np.max(theta_nm_range)-np.min(theta_nm_range))
            print("Angle", mean)
            print("Chromatic error", err)
        except ValueError:
            print("None found")
            print(mean)
            print(err)
            print("======")
        return theta_nm_range, wl_nm_range, mean, err

    def plotCherenkov(self):
        """Plot Angle against Wavelength and Scatter plot of intersection"""
        if not self.status['intersected']:
            print("Cherenkov angle has not been calculated, please use "
                  "calculateCherenkov(v=<speed>, direction=<[rho, z]>")
            return
        fig = plt.figure(figsize=(10,8))  # Cherenkov plot
        fig1 = plt.figure(figsize=(10,8))  # 3D scatter plot

        kz_axis = r"$k_z \,(m^{-1})$"
        k_rho_axis = r"$k_{\rho} \,(m^{-1})$"
 
        ax = fig.add_subplot(1,1,1)
        fig.tight_layout()
        ax.set_xlabel(r" Cherenkov Angle $\theta_c$ (rad)")
        ax.set_ylabel(r"Wavelength $\lambda$ (m)")
        ax.set_ylim([0,600])  # wavelength range
        ax.set_title("Wavelength against Cherenkov Angle (+ve) \n Derived from 2D Dispersion")

        ax1 = fig1.add_subplot(1,1,1, projection='3d')
        ax1.set_title("Intersection Points Between Electron Plane \n and 2D Dispersion")
        ax1.set_xlabel(k_rho_axis)
        ax1.set_ylabel(kz_axis)
        ax1.set_zlabel(r"Frequency (Hz)")

        for band in self.data['default']:
            print("Band", band + ":")
            th = self.data['default'][band]['cherenkov']['angle']
            th_pos = self.data['default'][band]['cherenkov']['pos']['angle']
            wl = self.data['default'][band]['cherenkov']['wavelength']
            kz = self.data['default'][band]['cherenkov']['kz']
            k_rho = self.data['default'][band]['cherenkov']['k_rho']
            f = self.data['default'][band]['cherenkov']['frequency']
            wl = np.array(wl)
            ax.plot(th, wl*1.e9, linestyle='None', marker='o', color='black')
            ax.set_xticks(np.arange(0,0.4+0.004, 0.05))  # angle
            ax.set_xlim([0, max(th_pos)+0.05])
            global_max = max([np.max(kz), np.max(k_rho)])
            ax1.set_ylim([-global_max, global_max])
            ax1.set_xlim([-global_max, global_max])
            ax1.invert_xaxis()  # reversing for viewing ease
            ax1.scatter(k_rho, kz, f, color='black')

        fig.savefig("cherenkov.png", bbox_inches='tight')
        fig1.savefig("intersection.png", bbox_inches='tight')
        fig.show()
        fig1.show()
        plt.close()

    def plot3d(self, mode='surface'):
        """Plot dispersion"""
        if not self.status['reflected']:
            print("Reflecting")
            self.reflect()
        if not self.status['interpolated']:
            print("Interpolating")
            self.interpolate()
        print("Plotting")
        fig = plt.figure(figsize=(12,9))

        for i, band in enumerate(self.data['default']):
            print(self.data.keys())
            print(self.data['default'][band].keys())
            mf = self.data['default'][band]['mf']
            m_rho = self.data['default'][band]['mi']
            mz = self.data['default'][band]['mj']

            ax = fig.add_subplot(1,1,1, projection='3d')
            global_max = np.max([np.max(m_rho), np.max(mz)])
            global_min = np.min([np.min(m_rho), np.min(mz)])
            ax.set_xlim([global_min, global_max])
            ax.set_ylim([global_min, global_max])
            ax.auto_scale_xyz([np.min(m_rho), np.max(m_rho)], \
                [np.min(mz), np.max(mz)], [np.min(mf),np.max(mf)])

            ax.set_title(r"Band "+str(i+1)+r" 3D Model Dispersion")
            ax.set_xlabel(r"Wavevector in $(k_x,k_y)$, $k_{\rho}$ $(m^{-1})$")
            ax.set_ylabel(r"Wavevector $k_z$ $(m^{-1})$")
            ax.set_zlabel(r"Frequency $(Hz)$")

            ax.invert_xaxis()
            print('rho', np.max(m_rho), 'z', np.max(mz))
            if mode == 'surface':
                surf = ax.plot_surface(m_rho, mz, mf, cmap=cm.bwr,
                                       linewidth=0, antialiased=False)
                #surf = ax.plot_surface(m_rho, mz, m_rho*3.e8, cmap=cm.bwr,
                #                       linewidth=0, antialiased=False)
            elif mode == 'scatter':
                ax.scatter(self.data['1']['k_rho'], self.data['1']['kz'], self.data['1']['frequency'])
                # ax.scatter(m_rho, mz, mf)
            elif mode == 'eplane':
                plane = ax.plot_surface(m_rho, mz, mz*0.999*3.e8, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
               
            #ax.set_zlim([np.min(mf),np.max(mf)])
            fig.savefig("dispersion"+band+".png", bbox_inches='tight')
            fig.show()
            plt.close()

    def compare_sio2(self, ratio_3d=0.106, index="sio2", \
        filename=None, modelname=None, n_lim=None):
        if not self.status['intersected']:
            print("Cherenkov angle has not been calculated, please use "
                  "calculateCherenkov(v=<speed>, direction=<[rho, z]>")
            return
        #    ratio_2d = (np.pi*(0.45*(3**0.5))**2)/(3*3**0.5/2)
        # volume ratio z direction
        # if ratio_3d is None:
        #     ratio_3d = 100./250.
        for band in self.data['default']:
            wl = self.data['default'][band]['cherenkov']['wavelength']
            th = self.data['default'][band]['cherenkov']['angle']
            effective.compare_medium(th, wl, ratio_3d, index=index,
                                    band=band, filename=filename,
                                    modelname=modelname, n_lim=[1.035,1.1])

    def photon_yield(self, beta=0.999, L=1.e-6, wl_range=[250.e-9, 500.e-9], \
                    root='default', band='0'):
        # raise NotImplementedError
        theta = self.data['default'][band]['cherenkov']['angle']

        f = self.data['default'][band]['cherenkov']['frequency']
        _, theta = self.wl_cut(root, band, wl_range=wl_range)
        _, f = self.wl_cut(root, band, wl_range, 'frequency') # TODO: move
        # print(theta)
        # print('============')
        # print(f)
        n_p = photon_yield.compute(theta=theta, f=f, beta=0.999,
                                  L=1.e-3, n=None)
        if 'yield' not in list(self.data[root][band]):
            self.data['default'][band]['yield'] = {
                'range': [],
                'L': [],
                'n_photons': []
            }
        self.data['default'][band]['yield']['range'].append(wl_range)
        self.data['default'][band]['yield']['L'].append(L)
        self.data['default'][band]['yield']['n_photons'].append(n_p)

    def wl_cut(self, a='default', band='0', 
              wl_range=[0.,1e10], param_key=None, sign=1):
        """Take cut of data based on wavelength range. Default behaviour
        removes negative angles"""
        wl = self.data[a][band]['cherenkov']['wavelength']
        theta = np.array(self.data[a][band]['cherenkov']['angle'])
        if param_key is not None:
            # print(self.data['default'][band]['cherenkov'])
            print('cutting for', param_key)
            param = self.data[a][band]['cherenkov'][param_key]
        else:
            param = theta
        wl_nm_range = []
        param_nm_range = []
        for i, w in enumerate(wl):
            if w < wl_range[1] and w > wl_range[0] and sign*theta[i]>0:
                wl_nm_range.append(w)
                param_nm_range.append(param[i])
        return wl_nm_range, param_nm_range
