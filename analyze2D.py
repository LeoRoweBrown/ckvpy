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

    def cropped_plot(self, filename=None, modelname=None, band=0, a_i=0,\
                    key=None, wl_range=[250e-9, 500e-9]):
        """Plot wavelength against angle in 250-500nm wavelength range for a_ith unit cell size
        !!Untested at the moment!!
        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of different models
        v===If more than one value of unit cell is available===v
        band (int): Band for which data is plotted
        a_i (int): Index of unit cell size, i.e. a_ith element of file_suffix gives unit cell size 'a'
        """
        #raise NotImplementedError
        if key is None:
            a = list(self.data.data_dict)[a_i]
        #print(a)
        wl1, th1, i = self._interp_angle(250e-9, a, band)
        wl2, th1, j = self._interp_angle(500e-9, a, band)
        #j += 1 # get j for wavelength > 500nm
        fig = plt.figure(figsize=(10,8))
        wl = self.data.data_dict[a][band]['wavelength'][i-1:j+1] # TODO: key error here
        angle = self.data.data_dict[a][band]['angle'][i-1:j+1]

        ax = fig.add_subplot(111)
        wl = np.array(wl)*1e9
        ax.plot(angle, wl, color='black', marker='o', markersize=5)
        title = (r"Saturated Cherenkov Angle against Wavelength"
                r"(Cropped to 250-500nm)")
        if modelname is None: modelname = self.path
        title += "\n (" + modelname + ")"
        ax.set_title(title)
        ax.set_ylabel(r"Wavelength (nm)")
        ax.set_xlabel(r"Saturated Cherenkov Angle $\theta_c$ (rad)")

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def full_plot(self, filename=None, modelname=None, \
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

    def a_plot(self, filename=None, modelname=None, band='0'):
        """Plot 'a' against Cherenkov angle and chromatic error
        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of different models
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

    def close_plots(self):
        plt.close()

    def compare_sio2(self, ratio_2d=0.1, modelname='', filename='', 
                     index="sio2", bands=[None], a_s=[None], 
                     wl_range=[250.e-9,500.e-9], n_lim=[1.02,1.1]):
        """Compare Cherenkov behaviour in simulation to predicted from
        Maxwell Garnett and plot with tools.effective.compare_medium()
        ratio_2d (float): Volume ratio between air and SiO2
        modelname (str): Used in title of graph to identify model
        filename (str): Used to decide filename (has index of a and 
            band appended, i.e. for a = 1e-7, 2e-7 index of a for 1e-7 is 0)
        bands (list[int]): List of bands in plot and save. If [None] do all
        a_s (list[str]): List of a (keys) to plot and save. If [None] do all
        wl_range (list[float]): Wavelength range to compare average refractive
            indices of theory and data 
        """
        self.data.calculate_n_eff()  # calculate n inside crystal using data
        for a_i, a in enumerate(self.data.data_dict):
            if a not in a_s and a_s[0] is not None:
                continue
            modelname_a = modelname + r"$a=$" + a + "m"

            for band in self.data.data_dict[a]:
                print(band)
                n_data = self.data.data_dict[a][band]['n_eff']
                wl_in = self.data.data_dict[a][band]['wl_in']
                th_in = self.data.data_dict[a][band]['th_in']
                if int(band) not in bands and bands[0] is not None:
                    print('Ignoring band', band)
                    continue
                print(filename+'a'+str(a_i)+'band')
                n_mg, _ = effective.compare_medium(
                    n_data, th_in, wl_in, ratio_2d, index=index, band=band, 
                    n_lim=n_lim, beta=self.beta, modelname=modelname_a,
                    filename=filename+'a'+str(a_i)+'band')
                self.data.data_dict[a][band]['neff_mg'] = n_mg.tolist()
                # find n where wl_range[0] < wavelength < wl_range[1]
                for r in range(len(wl_range)): # lower and upper range e.g.
                                               # wl_range = [250e-9, 500.e-9]
                    i = 0
                    w = 0.
                    # loop stops once the data point w > wl_range[r]
                    while w < wl_range[r] and i < len(wl_in):
                        w = wl_in[i]
                        if i+1 >= len(wl_in):
                            print("could not find", wl_range[r])
                            print("using", wl_in[i])
                            j = i 
                        else:
                            wl1_diff = wl_in[i+1] - wl_range[r]
                            wl2_diff = wl_range[r] - wl_in[i]
                            # decide data point by its proximity to
                            # wl_range[r]
                            if abs(wl1_diff) > abs(wl2_diff):
                                j = i
                            else:
                                j = i+1
                        i += 1  # next data point
                    if r == 0:
                        i1 = j
                    else:
                        i2 = j+1
                n_mg_av = np.mean(n_mg[i1:i2])  # average maxwell garnett n
                n_data_av = np.mean(n_data[i1:i2])  # average n from data
                n_mg_err = np.std(n_mg[i1:i2])/(i2-i1)**0.5  # standard error
                n_data_err = np.std(n_data[i1:i2])/(i2-i1)**0.5
                self.data.data_dict[a][band]['n_data_mean'] = \
                    [n_data_av, n_data_err]
                self.data.data_dict[a][band]['n_mg_mean'] = \
                    [n_mg_av, n_mg_err]
                print('a:', a, 'band:', band, 'n_data', n_data_av, '+-', \
                    n_data_err)
                print('a:', a, 'band:', band, 'n_mg', n_mg_av, '+-', \
                    n_mg_err)

    def photon_yield(self, beta=0.999, L=100.e-6, wl_range=[250.e-9, 500.e-9], \
                    root='default', band='0'):
            # raise NotImplementedError
            if root == 'default':
                # takes first key as root if default
                # this is already 'default' in 3d case
                root = list(self.data.data_dict)[0]
            theta = self.data.data_dict[root][band]['angle']
            f = self.data.data_dict[root][band]['frequency']
            wl, theta = self.wl_cut(root, band, wl_range)
            wl, f = self.wl_cut(root, band, wl_range, 'frequency')
            n_p = photon_yield.compute(theta=theta, f=f, beta=0.999,
                                    L=L, n=None)
            if 'yield' not in list(self.data.data_dict[root][band]):
                self.data.data_dict[root][band]['yield'] = {
                    'range': [],
                    'L': [],
                    'n_photons': []
                }
            self.data.data_dict[root][band]['yield']['range'].append(wl_range)
            self.data.data_dict[root][band]['yield']['L'].append(L)
            self.data.data_dict[root][band]['yield']['n_photons'].append(n_p)