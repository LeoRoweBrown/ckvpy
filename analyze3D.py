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
from ckvpy.tools.analysis3d import dataAnalysis3D

__all__ = ['Analyze3D']

class Analyze3D():
    """Intersect 2D band structure with electron plane. Can be used for 3D 
    data or 2D data, but 3D data must be confined to a 2D plane which at
    the moment is constrained to just planes in k_rho,kz where 
    k_rho = norm(kx,ky). Realistically in COMSOl this means only kz,kx and
    kz,ky because of the way bands are found: a direction of k is chosen
    and increase |k|. Always sorted by band (band is the root key)""" 
    def __init__(self, datafile, symmetry=4, headers=['skip', 'band', 'skip',
                 'frequency', 'kx', 'ky', 'kz', 'n', 'skip'], add_zero=False,
                 ndim=3, resolution=100, interpolation_method='cubic'):

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
        
        data_loader = Bzone2D(
            datafile, headers=headers, ndim=ndim, symmetry=symmetry, 
            interpolation_method='cubic', resolution=resolution, 
            add_zero=False)
        self._init_analysis(data_loader)


    def _init_analysis(self, data_loader):
        """Pass data dictionary to dataAnalysis object, access raw data with
        self.data.data_dict[keys]"""
        self.data = dataAnalysis3D(data_loader.data) 
        # TODO: now replace every self.calc_err 
        # with data.calc_err etc. and self.data=>self.data.data_dict (avoid this?):
        # __getitem__(self, key):
        #     return object.__getattribute(self.data, key)

    def plot_range(self):
        """Plot effect of wavelength range on chromatic error and angle"""
        for band in self.data.data_dict['default']:
            wl_low_a = []
            mean_a = []
            err_a = []
            theta = self.data.data_dict['default'][band]['angle']
            # self.sort_data('wavelength', subkeys=['cherenkov'])
            wl = \
                np.array(self.data.data_dict['default'][band]['wavelength'])
            # print(wl)
            # print(theta)
            for i, wl_low in enumerate(range(250,401,50)):
                print(wl_low)
                wl_r = [wl_low*1.e-9, 500.e-9]
                _, _, mean_t, err_t = \
                    self.data.calc_err(wl_range, 'default', band, sign=1)
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

    def plot_cherenkov(self):
        """Plot Angle against Wavelength and Scatter plot of intersection"""
        if not self.data.status['intersected']:
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

        for band in self.data.data_dict['default']:
            print("Band", band + ":")
            th = self.data.data_dict['default'][band]['angle']
            wl = self.data.data_dict['default'][band]['wavelength']
            kz = self.data.data_dict['default'][band]['kz']
            k_rho = self.data.data_dict['default'][band]['k_rho']
            f = self.data.data_dict['default'][band]['frequency']
            # wl, f_cut = self.data.wl_cut(
            #     root='default', band='0', wl_range=[0.,1e10])
            # wl, th = self.data.wl_cut(
            #     root='default', band='0', wl_range=[0.,1e10],
            #     param_key = 'angle')
            wl = np.array(wl)
            print(max(th))
            ax.plot(th, wl*1.e9, linestyle='None', marker='o', color='black')
            ax.set_xticks(np.arange(0,0.4+0.004, 0.05))  # angle
            ax.set_xlim([0, 0.4+0.05])
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

    def plot_3d(self, mode='surface'):
        """Plot dispersion"""
        # if not self.data.status['reflected']: # from old structure of code
        #     print("Reflecting")
        #     self.data.reflect()
        # if not self.data.status['interpolated']:
        #     print("Interpolating")
        #     self.interpolate()
        print("Plotting")
        fig = plt.figure(figsize=(12,9))

        for i, band in enumerate(self.data.data_dict['default']):
            print(self.data.data_full.keys())
            print(self.data.data_full['default'][band].keys())
            mf = self.data.data_full['default'][band]['mf']
            m_rho = self.data.data_full['default'][band]['mi']
            mz = self.data.data_full['default'][band]['mj']

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
                ax.scatter(self.data.data_full['default']['0']['k_rho'], \
                    self.data.data_full['default']['0']['kz'], \
                    self.data.data_full['default']['0']['frequency'])
                # ax.scatter(m_rho, mz, mf)
            # elif mode == 'eplane':
            #     plane = ax.plot_surface(m_rho, mz, mz*0.999*3.e8, \
            #                 cmap=cm.coolwarm, linewidth=0, antialiased=False)
               
            #ax.set_zlim([np.min(mf),np.max(mf)])
            fig.savefig("dispersion"+band+".png", bbox_inches='tight')
            plt.show()
            plt.close()

    def compare_sio2(self, ratio_3d=0.106, index="sio2", \
        filename=None, modelname=None, n_lim=None):
        """TODO not used, overrridend and deprecated. Remove"""
        raise NotImplementedError
        if not self.data.status['intersected']:
            print("Cherenkov angle has not been calculated, please use "
                  "calculateCherenkov(v=<speed>, direction=<[rho, z]>")
            return
        #    ratio_2d = (np.pi*(0.45*(3**0.5))**2)/(3*3**0.5/2)
        # volume ratio z direction
        # if ratio_3d is None:
        #     ratio_3d = 100./250.
        for band in self.data.data_dict['default']:
            n_data = self.data.data_dict['default'][band]['n_eff']
            wl_in = self.data.data_dict['default'][band]['wl_in']
            th_in = self.data.data_dict['default'][band]['th_in']
            effective.compare_medium(
                n_data, th_in, wl_in, ratio_3d, 
                index=index, band=band, filename=filename, 
                modelname=modelname, n_lim=[1.035,1.1]
                )

    def compare_sio2(self, ratio=0.106, index="sio2", \
    filename=None, modelname=None, \
    n_lim=None, roots=None, bands=None):
        """Compare expected refractive index/Cherenkov angle from 
        Maxwell-Garnett formula to data from simulation. Analysis is valid
        INSIDE the crystal, so wavelength derived from k not c/f
        Params:
            ratio (float): ratio of dielectric to air
            index (str): material refractive index in ./index/<index>.txt
                used to calculate expected index from effective medium.
            filename (str): prefix for file names of form 
                filename_a_1e-7_b_0.png
            modelname (str): modelname used in title of graph 
            n_lim: n plot range
            roots (list (str)): root keys to plot for
            bands (list (str)): bands to plot for
        """
        for root in self.data.data_dict:
            if root not in roots and roots is not None:
                continue

            for band in self.data.data_dict[root]:
                if band not in bands and bands is not None:
                    continue

                if 'n_eff' not in self.data.data_dict[root][band]:
                    # calculate n inside crystal using data
                    self.data.calculate_n_eff()  
                if 'n_mg' not in self.data.data_dict[root][band]:
                    if ratio is None:
                        print("Supply volume ratio or run "
                              "data.calculate_n_mg first")
                        return
                    # calculate n using MG theory
                    self.data.calculate_n_mg(ratio, index)

                wl_in = self.data.data_dict[root][band]['wl_in']
                th_in = self.data.data_dict[root][band]['th_in']
                n_data = self.data.data_dict[root][band]['n_eff']
                ind = np.argsort(wl_in)
                wl_in = np.array([wl_in[i] for i in ind]) # use data.sort_data() 
                                                        # instead?
                th_in = np.array([th_in[i] for i in ind]) # unused at the moment

                n_mg = self.data.data_dict[root][band]['n_mg']
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111)
                ax.plot(wl_in*1e9, n_data,
                    label="Simulation",  linestyle='None', color='black', marker='o',\
                    markersize=6)
                ax.plot(wl_in*1e9, n_mg, label="Effective medium theory "
                    "(Maxwell-Garnett)",
                    color='black', linestyle='--', marker='None', markersize=6)
                ax.set_xticks(np.arange(np.min(wl_in)-100, 1000+100, 100))
                global_max = max([np.max(n_mg), np.max(n_data)])
                global_min = min([np.min(n_mg), np.min(n_data)])

                if n_lim is None:
                    n_lim = global_min, global_max
                ax.set_ylim(n_lim)
                ax.set_xlim([200,600])  # Malitson SiO2 only valid from 200nm

                title = ("Effective Index Comparison Between Theory and "
                        "Simulation for \n" + r"($a=$"+root+r"$m$) (Band " + \
                        str(int(band)+1) + ")")
                if modelname is not None:
                    title += " (" + modelname + ")"
                ax.set_title(title)
                ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
                ax.set_ylabel(r"Refractive index $n_{eff}$")
                ax.legend()
                if filename is None:
                    fig.savefig("3d_untitled_effective_index_a_"+\
                        str(root)+"b_"+str(band)+".png")
                else:
                    fig.savefig(filename+"_a_"+\
                        str(root)+"b_"+str(band)+".png")
                plt.close()