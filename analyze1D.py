import numpy as np
import scipy
import cmath
import csv
from matplotlib import pyplot as plt
import itertools
from ckvpy.tools.csvloader import CSVLoader
import ckvpy.tools.effective as effective

class Analyze1D(CSVLoader):
    def __init__(self, datafile, file_suffix = 'undefined', \
        headers = ['band', 'k', 'frequency', 'wavelength', 'angle', 'a'],
        sort_by = 'a'):
        """Analyses  Chernekov data (in a single k direction)
        for a 2D model from a CSV file of wavelengths and angles 
        Inherits from CSVLoader class in tools.py

        """
        header_list = \
        ['band', 'k', 'frequency', 'wavelength', 'angle', 'a', 'l', 'skip']
        np.set_printoptions(precision=3)
        for header in headers:
            if header not in header_list:
                raise ValueError("Invalid header supplied, use one of",
                                 header_list)
        self.mpl_lines = itertools.cycle(["-",":","--","-.",])
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams['font.size'] = 14

        super(Analyze1D, self).__init__(
            datafile=datafile, file_suffix=file_suffix, headers=headers, 
            sort_by=sort_by)

    def load_mat(self, matrixfile):
        try:
            matrix = np.loadtxt(matrixfile, ndmin=2)
            for row, a in enumerate(self.data):
                self.data[a]['cherenkov'] = matrix[row]
        except IOError:
            raise IOError(
                "No such file, generate data using get_angle with dump=True "
                "and filename=<filename> "
                "with raw data specified with CherenkovData(data=<filename>)"
                )

    def _interp_angle(self, wavelength, a, band=0):
        """Interpolate between two points of data to find angle at desired 
        wavelength for given band e.g. wavelength=250nm -> interpolate angle
        between between points 240nm and 260nm
        
        Params:
        wavelength (float): desired wavelength (in m or same unit as data) at
            which to solve for angle 
        a (str): key for unit cell size in data dictionary
        """
        i = 0
        # make sure in ascending order for line 
        # "while float(wl[i]) < wavelength"
        self.sort_data('wavelength') 
        wl = self.data[a]['wavelength'][band]
        th = self.data[a]['angle'][band]
        
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
                  band=0):
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
        self.a_matrix = np.array([]).reshape(0,5) 
        for a in self.data:
            print("Finding angle for a =", a)
            wl1, th1, i1 = self._interp_angle(wl_range[0], a, band)
            wl2, th2, i2 = self._interp_angle(wl_range[1], a, band)
            th = self.data[a]['angle'][band]
            average = np.average(th[i1:i2])
            rnge = abs(th1-th2)
            array = np.array([[wl1, wl2, average, rnge, float(a)]])
            self.data[a]['cherenkov'] = array.tolist()  # json friendly

            if filename is not None: # TODO: deprecated
                try:
                    matrix = np.loadtxt(filename, ndmin=2) 
                    if min([float(a) - line[-1] for line in matrix]) < 1e-12:

                        # duplicate so make array empty
                        array=np.array([]).reshape(0,5) 
                        print("Duplicates found, not replacing")
                except:
                    #force 0,5 shape for single array in file
                    matrix = np.array([]).reshape(0,5)
                np.savetxt(filename, np.concatenate((matrix, array), axis=0))

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
        wl = self.data[a]['wavelength'][band][i-1:j+1] # TODO: key error here
        angle = self.data[a]['angle'][band][i-1:j+1]

        ax = fig.add_subplot(111)
        wl = np.array(wl)*1e9
        ax.plot(angle, wl, color='black', marker='o', markersize=5)
        title = r"Saturated Cherenkov Angle against Wavelength (Cropped to 250-500nm)"
        if modelname is None: modelname = self.path
        title += "\n (" + modelname + ")"
        ax.set_title(title)
        ax.set_ylabel(r"Wavelength (nm)")
        ax.set_xlabel(r"Saturated Cherenkov Angle $\theta_c$ (rad)")

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
        else:
            fig.show()
    
    def full_plot(self, filename=None, modelname=None, a_i=0, key=None, dump=False):
        """Plot wavelength against angle in full wavelength and angle range for a_ith unit cell size

        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of different models
        band (int): Band for which data is plotted
        a_i (int): Index of unit cell size, i.e. a_ith element of file_suffix gives unit cell size 'a'
        key (str): Alternative to a_i, supply key directly
        """
        if key is None:
            a = list(self.data)[a_i]
        fig = plt.figure(figsize=(10,8))
        max_th = 0
        ax = fig.add_subplot(111)
        #print(self.data[a]['angle'])
        for b, (w, th) in enumerate(zip(self.data[a]['wavelength'], self.data[a]['angle'])):
            w = np.array(w)*1e9
            th = np.ma.array(th) # mask angle of 0 to prevent line being drawn to it
            #mask values below 1e-3
            th_masked = np.ma.masked_where(th < 1e-3 , th)

            ax.plot(th_masked, w, marker='o', markersize=5, linestyle=next(self.mpl_lines),\
            color='black', label="band "+str(b+1) ) #linestyle='None', #color='black'
            ax.legend()
            #print(a)
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

    def a_plot(self, filename=None, modelname=None, band=0):
        """Plot 'a' against Cherenkov angle and chromatic error
        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of different models
        """
        # matrix of data
        lists = [ np.array([None]*len(self.data)) for _ in self.data]
        wl1, wl2, angle, err, a, = \
            lists[0], lists[1], lists[2], lists[3], lists[4]

        for i, a_key in enumerate(self.data):
            wl1[i], wl2[i], angle[i], err[i], a[i] = \
                self.data[a_key]['cherenkov'][band]
            print(self.data[a_key]['cherenkov'][band])
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

    def compare_sio2(self, ratio_2d, modelname, filename, index="sio2",
                    bands=[None], a_s=[None]):
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
            wl = self.data[a]['wavelength']
            th = self.data[a]['angle']
            for b, wl_i in enumerate(wl):
                if b not in bands and bands[0] is not None:
                    continue
                th_i = th[b]
                effective.compare_medium(
                    th_i, wl_i, ratio_2d, index=index, band=b,
                    modelname=modelname_a, filename=filename+str(a_i))


