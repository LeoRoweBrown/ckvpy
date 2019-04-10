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

__all__ = ['Analyze2D']

class Analyze2D(CSVLoader):
    def __init__(self, datafile, file_suffix = 'undefined', \
        headers = ['a', 'band', 'k', 'frequency', 'wavelength', 'angle'],
        sort_by = 'a'):
        """Analyses  Chernekov data (in a single k-in-plane direction)
        for a 2D model from a CSV file of wavelengths and angles 
        Inherits from CSVLoader class in tools.py

        """
        # may change structure to include header_list only in base class
        header_list = \
        ['band', 'k', 'frequency', 'wavelength', 'angle', 'a', 'l', 'skip']
        # np.set_printoptions(precision=3)
        for header in headers:
            if header not in header_list:
                raise ValueError("Invalid header supplied, use one of",
                                 header_list)
        self.mpl_lines = itertools.cycle(["-",":","--","-.",])
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams['font.size'] = 14

        super(Analyze2D, self).__init__(
            datafile=datafile, file_suffix=file_suffix, headers=headers, 
            sort_by=sort_by)

    def _interp_angle(self, wavelength, a, band='0'):
        """Interpolate between two points of data to find angle at desired 
        wavelength for given band e.g. wavelength=250nm -> interpolate angle
        between between points 240nm and 260nm
        
        Params:
        wavelength (float): desired wavelength (in m or same unit as data) at
            which to solve for angle 
        a (str): key for unit cell size in data dictionary
        """
        i = 0
        band = str(band)
        # make sure in ascending order for line 
        # "while float(wl[i]) < wavelength"
        self.sort_data('wavelength') 
        wl = self.data[a][band]['wavelength']
        th = self.data[a][band]['angle']
        
        while float(wl[i]) < wavelength:
            i += 1 # condition stops when data point is greater than
                   # wavelength, so look at i and i-1 for correct range
            if i > len(wl):
                raise ValueError("Failed to find angles for wavelength =",
                                 wavelength,
                                 ". Check that it exists in data.")
        if wl[i-1] > wavelength: 
            # wl[i-1] should be the value smaller than desired wavelength
            print("Wavelength too small! Changing", wavelength, "to", wl[i-1])
            print("Dataset doesn't seem to match desired wavelength range, "
                  "expect strange results (it is likely that the angles "
                  "found will not be anywhere near the vertical line")
            wavelength = wl[i-1]
        elif wl[i] < wavelength: 
            # wl[i] should be the value larger than desired wavelength
            print("Wavelength too large! Changing", wavelength, "to", wl[i])
            raise Warning("Dataset doesn't seem to match desired wavelength "
                          "range, expect strange results (it is likely that "
                          "the angles found will not be anywhere near the "
                          "vertical line")
            wavelength = wl[i]

        if (wl[i]-wl[i-1]) < 1.e-15*(th[i]-th[i-1]): # avoid division by zero
            angle = (th[i]+th[i-1])/2 # take average if angle is multivalued
                                      # if wl1 and wl2 are the same
        else:
            angle = (th[i]-th[i-1])/(wl[i]-wl[i-1])*(wavelength-wl[i-1]) \
                + th[i-1] # grad*(wavelength1-wavelength0) + angle0
        print("found", (wavelength,angle), "between", (wl[i], th[i]), \
              "and", (wl[i-1], th[i-1]) )
        return wavelength, angle, i

    def find_angle(self, wl_range=[250.0e-9, 500.0e-9], filename=None,\
                  band='0'):
        """Get Cherenkov angles for wavelength range wl_range
        
        Params:
        filename (str): name of file that 'a', average Cherenkov
            angle, chromatic error in 250-500nm wavelength range is stored
        
        Returns:
        Single mode (see csvloader.py):
            Cherenkov angle average and range (when in single mode)
        Merged or split mode:
            None
        """
        band = str(band)
        for a in self.data:
            for band in self.data[a]:
                print("Finding angle for a =", a, "band", band)
                wl1, wl2, average, rnge = self.calc_err(wl_range, a=a, band=band)
                array = np.array([wl1, wl2, average, rnge, float(a)])
                self.data[a][band]['cherenkov'] = array.tolist()  # json friendly
                # self.data[a][band][str(wl1)+'-']

        # print(average, rnge)
        if self.format is 'single':
            return average, rnge
        return # dont return average and range if computed for multiple values of 'a', these are stored in file.

    def cropped_plot(self, filename=None, modelname=None, band=0, a_i=0,\
                    key=None, wl_range=[250e-9, 500e-9]):
        """Plot wavelength against angle in 250-500nm wavelength range for a_ith unit cell size

        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of different models
        v===If more than one value of unit cell is available===v
        band (int): Band for which data is plotted
        a_i (int): Index of unit cell size, i.e. a_ith element of file_suffix gives unit cell size 'a'
        """
        #raise NotImplementedError
        if key is None:
            a = list(self.data)[a_i]
        #print(a)
        wl1, th1, i = self._interp_angle(250e-9, a, band)
        wl2, th1, j = self._interp_angle(500e-9, a, band)
        #j += 1 # get j for wavelength > 500nm
        fig = plt.figure(figsize=(10,8))
        wl = self.data[a][band]['wavelength'][i-1:j+1] # TODO: key error here
        angle = self.data[a][band]['angle'][i-1:j+1]

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
        else:
            fig.show()
    
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
            a = list(self.data)[a_i]
        if bands is None:
            bands = list(self.data[a])
        else:
            bands = [str(b) for b in bands]
        #print(self.data[a][band]['angle'])
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        for band in self.data[a]:
            if band not in bands: 
                continue
            print(band)
            w = self.data[a][band]['wavelength']
            th = self.data[a][band]['angle']
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
        else:
            fig.show()

    def a_plot(self, filename=None, modelname=None, band='0'):
        """Plot 'a' against Cherenkov angle and chromatic error
        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of different models
        """
        # matrix of data
        band = str(band)
        lists = [ np.array([None]*len(self.data)) for _ in self.data]
        wl1, wl2, angle, err, a, = \
            lists[0], lists[1], lists[2], lists[3], lists[4]

        for i, a_key in enumerate(self.data):
            wl1[i], wl2[i], angle[i], err[i], a[i] = \
                self.data[a_key][band]['cherenkov']
            print(self.data[a_key][band]['cherenkov'])
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
        fig.show()
        if filename is not None:
            print("Saving as", filename)
            fig.savefig(filename, bbox_inches='tight')
        else:
            fig.show()

    def compare_sio2(self, ratio_2d=0.9, modelname=None, filename=None,
                     index="sio2", bands=[None], a_s=[None]):
        """Compare Cherenkov behaviour in simulation to predicted from
        Maxwell Garnett and plot with tools.effective.compare_medium()
        ratio_2d (float): Volume ratio between air and SiO2
        modelname (str): Used in title of graph to identify model
        filename (str): Used to decide filename (has index of a and 
            band appended, i.e. for a = 1e-7, 2e-7 index of a for 1e-7 is 0)
        bands (list[int]): List of bands in plot and save. If [None] do all
        a_s (list[str]): List of a (keys) to plot and save. If [None] do all
        """
        for a_i, a in enumerate(self.data):
            if a not in a_s and a_s[0] is not None:
                continue
            modelname_a = modelname + r" $a=$" + a + "m"

            for band in self.data[a]:
                wl = self.data[a][band]['wavelength']
                th = self.data[a][band]['angle']
                if int(band) not in bands and bands[0] is not None:
                    continue
                effective.compare_medium(
                    th, wl, ratio_2d, index=index, band=band,
                    modelname=modelname_a, filename=filename+str(a_i))

    def photon_yield(self, wl_range=[250.e-9, 500.e9]):
        raise NotImplementedError
        for a in self.data:
            for band in self.data[a]:
                theta = self.data[a][band]['angle']
                wl = self.data[a][band]['wavelength']
                f = self.data[a][band]['frequency']
                theta_cut, wl_cut, mean, err = \
                    self._calc_err(theta, wl, wl_range)
                photon_yield.compute(theta_cut, f)
    
    def save_table(self, filename):
        matrix = np.empty((0,5))
        for a in self.data:
            for b in self.data[a]:
                line = np.array(self.data[a][b]['cherenkov'])
                matrix = np.vstack((matrix, line))
        print(matrix)
        np.savetxt(filename, matrix)
            
    def photon_yield(self, beta=0.999, L=100.e-6, wl_range=[250.e-9, 500.e-9], \
                    root='default', band='0'):
            # raise NotImplementedError
            if root == 'default':
                root = list(self.data)[0]
            theta = self.data[root][band]['angle']
            f = self.data[root][band]['frequency']
            theta, f, wl = self.wl_cut(root, band, 'frequency', wl_range)
            n_p = photon_yield.compute(theta=theta, f=f, beta=0.999,
                                    L=L, n=None)
            if 'yield' not in list(self.data[root][band]):
                self.data[root][band]['yield'] = {
                    'range': [],
                    'L': [],
                    'n_photons': []
                }
            self.data[root][band]['yield']['range'].append(wl_range)
            self.data[root][band]['yield']['L'].append(L)
            self.data[root][band]['yield']['n_photons'].append(n_p)