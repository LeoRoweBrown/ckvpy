import numpy as np
import scipy
import cmath
import csv
import json
from matplotlib import pyplot as plt
import itertools
from ckvpy.tools.csvloader import CSVLoader
import ckvpy.tools.effective as effective
import ckvpy.tools.photon_yield as photon_yield
from ckvpy.tools.analysis import dataAnalysis

__all__ = ['Analyze2D']

class Analyze2D():
    """For plotting analysed data for case of 2D model, most actual analysis
    takes place in analysis.py which contains data analysis classes"""
    def __init__(self, datafile, file_suffix = 'undefined', \
        headers = ['a', 'band', 'k', 'frequency', 'wavelength', 'angle',
        'kx', 'ky', 'kz'], sort_by = 'a', beta=0.999):
        """Analyses  Chernekov data (in a single k-in-plane direction)
        for a 2D model from a CSV file of wavelengths and angles 
        Inherits from CSVLoader class in tools.py

        """
        # may change structure to include header_list only in base class
        header_list = \
        ['band', 'k', 'frequency', 'wavelength', 'angle', 'a', 'l', 'skip',
         'kx', 'ky', 'kz', 'n']
        # np.set_printoptions(precision=3)
        for header in headers:
            if header not in header_list:
                raise ValueError("Invalid header supplied, use one of ",
                                 header_list)
        self.mpl_lines = itertools.cycle(["-",":","--","-.",])
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams['font.size'] = 14

        data_loader = CSVLoader(datafile=datafile, file_suffix=file_suffix, 
            headers=headers, sort_by=sort_by)
        self._init_analysis(data_loader)
        self.path = data_loader.path
        self.beta = beta
    
    def _init_analysis(self, data_loader):
        """Pass data dictionary to dataAnalysis object"""
        self.data = dataAnalysis(data_loader.data)
        # TODO: now replace every self.calc_err 
        # with data.calc_err etc. 
    
    def plot_cherenkov(self, filename=None, modelname=None, \
        a_i=0, a=None, dump=False, bands=None):
        """Plot wavelength against angle in full wavelength and angle range 
        for a/a_ith unit cell size

        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of
            different models
        band (int): Band for which data is plotted
        a_i (int): Index of unit cell size, i.e. a_ith element of file_suffix
            gives unit cell size 'a'
        a (str): Alternative to a_i, supply key directly
        bands list[int/str]: Bands to plot, if None plot all (count from 0)
        """
        if a is None:
            a = list(self.data.data_dict)[a_i]
        if bands is None:
            bands = list(self.data.data_dict[a])
        else:
            bands = [str(b) for b in bands]
        #print(self.data.data_dict[a][band]['angle'])
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        for band in self.data.data_dict[a]:
            if band not in bands: 
                continue
            print(band)
            w = self.data.data_dict[a][band]['wavelength']
            th = self.data.data_dict[a][band]['angle']
            w = np.array(w)*1e9
            # mask angle of 0 to prevent line being drawn to it
            th = np.ma.array(th)
            #mask values below 1e-3
            th_masked = np.ma.masked_where(th < 1e-3 , th)
            ax.plot(th_masked, w, marker='o', markersize=5, 
                linestyle=next(self.mpl_lines), color='black',
                label="band "+str(int(band)+1) )
        ax.legend()
        title = r"Saturated Cherenkov Angle against Wavelength"
        if modelname is None: modelname = self.path
        title += "\n" + r"$a=$" + a + "m (" + modelname + ")"
        #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax.set_title(title)
        ax.set_ylabel(r"Wavelength (nm)")
        #print(np.arange(0, max(wl),max(wl)/10))
        #ax.set_yticks(np.arange(0.0, max(wl), max(wl)/10))
        ax.set_xticks(np.arange(0.0, 0.5+0.05, 0.05))
        ax.set_xlim([0,0.5])
        ax.set_ylim([0,1.e3])
        ax.set_yticks(np.arange(0,1.01e3,100))
        ax.set_xlabel(r"Saturated Cherenkov Angle $\theta_c$ (rad)")
        if filename is not None:
            print("Saving as", filename)
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_a(self, filename=None, modelname=None, band='0'):
        """Plot 'a' against Cherenkov angle and chromatic error
        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of different models
        band (str/int): Band to plot for
        """
        # matrix of data
        band = str(band)
        lists = [np.array([None]*len(self.data.data_dict)) \
            for _ in self.data.data_dict]
        wl1, wl2, angle, err, a, = \
            lists[0], lists[1], lists[2], lists[3], lists[4]

        for i, a_key in enumerate(self.data.data_dict):
            wl1[i], wl2[i], angle[i], err[i], a[i] = \
                self.data.data_dict[a_key][band]['cherenkov']
            print(self.data.data_dict[a_key][band]['cherenkov'])
        # exclude different wavelength ranges?
        #print(angle, err, a)
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(111)
        #ax.plot(a, angle)
        ax.errorbar(x=a, y=angle, yerr=err/2., color='black', capsize=5,\
            marker='o', markersize=5) # error divided by 2 for errorbar
        title = r"Saturated Cherenkov Angle Against Unit Cell Size"
        if modelname is None: modelname = self.path
        title += "\n for " + str(min(wl1)*1e9) + r"nm" + r"$< \lambda <$" + \
            str(max(wl2)*1e9) + r"nm" + " (" + modelname + ")"

        ax.set_title(title)
        ax.set_xlabel("Unit cell dimension")
        ax.set_xlabel(r"Unit cell dimension $a$ (m)")

        ax.set_ylabel(r"Average Saturated Cherenkov Angle $\theta_c$ (rad)")
        if filename is not None:
            print("Saving as", filename)
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def compare_sio2(self, ratio=None, index="sio2", \
    filename=None, modelname=None, \
    n_lim=None, beta=0.999, roots=[None], bands=[None]):
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
            bands (list (str)): bands to plot for"""
            
        for root in self.data.data_dict:
            if root not in roots and roots[0] is not None:
                continue

            for band in self.data.data_dict[root]:
                if band not in bands and bands[0] is not None:
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

                self.data.sort_data('wavelength')  # not needed?

                wl_in = np.array(self.data.data_dict[root][band]['wl_in'])
                th_in = np.array(self.data.data_dict[root][band]['th_in'])
                n_data = np.array(self.data.data_dict[root][band]['n_eff'])
                # ind = np.argsort(wl_in)
                # wl_in = np.array([wl_in[i] for i in ind]) # use data.sort_data() 
                                                        # instead?
                # th_in = np.array([th_in[i] for i in ind]) # unused at the moment
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
                ax.set_xlim([200,1000])  # Malitson SiO2 only valid from 200nm

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
                    fig.savefig("untitled_effective_index_a_"+\
                        str(root)+"b_"+str(band)+".png")
                else:
                    fig.savefig(filename+"_a_"+\
                        str(root)+"b_"+str(band)+".png")
                plt.close()