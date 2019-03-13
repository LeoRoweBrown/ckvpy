import numpy as np
import scipy
import cmath
import csv
from matplotlib import pyplot as plt
import itertools
from . import bases


class Analyze1D(bases.CSVLoader):
    def __init__(self, datafile, file_suffix = 'undefined', headers = ['band', 'k', 'wavelength', 'angle', 'a'], sort_by = 'a'):
        """datafile contains full Cherenkov vs wavelength behaviour, matrixfile just contains average angle vs
        unit cell size. Inherited from CSVLoader class in bases.py

        """
        plt.rcParams.update({'font.size': 14})
        np.set_printoptions(precision=3) # will this affect data saved to text?
        for header in headers:
            if header not in ['band', 'k', 'wavelength', 'angle', 'a', 'l', 'skip']:
                raise ValueError("Invalid header supplied, must be one of ['band', 'k', 'wavelength', 'angle', 'a', 'l']")
        self.mpl_lines = itertools.cycle(["-",":","--","-.",])

        return super(Analyze1D, self).__init__(datafile=datafile, file_suffix=file_suffix, headers=headers, sort_by=sort_by)

    def load_mat(self, matrixfile):
        try:
            self.a_matrix = np.loadtxt(matrixfile, ndmin=2)
        except IOError:
            raise IOError(
                "No such file, generate data using get_angle with dump=True and filename=<filename> "
                "with raw data specified with CherenkovData(data=<filename>)"
                )

    def _interp_angle(self, wavelength, a, band=0):
        """Interpolate between two points of data to find angle at desired wavelength for given band e.g. 
        wavelength=250nm -> interpolate angle between between points 240nm and 260nm
        
        Params:
        wavelength (float): desired wavelength (in m or same unit as data) at which to solve for angle 
        a (str): key for unit cell size in data dictionary
        """
        i = 0
        self.sort_data('wavelength') # make sure in ascending order for line "while float(wl[i]) < wavelength"
        wl = self.data[a]['wavelength'][band]
        th = self.data[a]['angle'][band]
        
        #sort_index = np.argsort(wl)
        #wl = [wl[ind] for ind in sort_index]
        #th = [th[ind] for ind in sort_index]

        while float(wl[i]) < wavelength: # data is organised with wavelength increasing large
            i += 1 # condition stops when data point is greater than wavelength, so look at i and i-1 for correct range
            if i > len(wl):
                raise ValueError("Failed to find angles for wavelength =", wavelength, ". Check that it exists in data.")
        if wl[i-1] > wavelength: # wl[i-1] should be the value smaller than desired wavelength
            print("Wavelength too small! Changing", wavelength, "to", wl[i-1])
            raise Warning("""Dataset doesn't seem to match desired wavelength range, expect strange results (it is likely
            that the angles found will not be anywhere near the vertical line""")
            wavelength = wl[i-1]
        elif wl[i] < wavelength: # wl[i] should be the value larger than desired wavelength
            print("Wavelength too large! Changing", wavelength, "to", wl[i])
            raise Warning("""Dataset doesn't seem to match desired wavelength range, expect strange results (it is likely
            that the angles found will not be anywhere near the vertical line""")
            wavelength = wl[i]


        if (wl[i]-wl[i-1]) < 1.e-15*(th[i]-th[i-1]): # avoid division by zero
            angle = (th[i]+th[i-1])/2 # take average if angle is multivalued for one wavelength (if wl1 and wl2 are the same)
        else:
            angle = (th[i]-th[i-1])/(wl[i]-wl[i-1])*(wavelength-wl[i-1]) + th[i-1] # grad*(wavelength1-wavelength0) + angle0
        print("found", (wavelength,angle), "between", (wl[i], th[i]), "and", (wl[i-1], th[i-1]) )
        return wavelength, angle, i

    def find_angle(self, filename=None, band=0):
        """Get Cherenkov angles for wavelength range 250-500nm
        
        Params:
        filename (str): name of file that 'a', average Cherenkov angle, chromatic error in 250-500nm wavelength range is stored
        
        Returns:
        Average Cherenkov angles for wavelength range 250-500nm [for single a or loop = False]
        """
        #th1, i = self._interp_angle(250e-9, band=band, a=a)
        #th2, j = self._interp_angle(500e-9, band=band, a=a)
        #average = np.average([th1, th2])
        #rnge = abs(th1-th2)
        for a in self.data:
            print("Finding angle for a =", a)
            wl1, th1, _ = self._interp_angle(250e-9, a, band)
            wl2, th2, _ = self._interp_angle(500e-9, a, band)
            average = np.average([th1, th2])
            rnge = abs(th1-th2)
            array = np.array([[wl1, wl2, average, rnge, float(a)]])
            if filename is not None:
                try:
                    matrix = np.loadtxt(filename, ndmin=2) #force 3,1 shape for single array in file
                    if min([float(a) - line[-1] for line in matrix]) < 1e-12:
                        array=np.array([]).reshape(0,5) # duplicate so make array empty (not tested)
                        print("Duplicates found, not replacing")
                    #print(matrix.shape)
                    #self.path[:-4]+"_a.txt"
                except:
                    matrix = np.array([]).reshape(0,5)
                np.savetxt(filename, np.concatenate((matrix, array), axis=0))
                self.a_matrix = np.concatenate((matrix, array), axis=0)
        if len(self.file_suffix) == 1: return average, rnge 
        return # dont return average and range if computed for multiple values of 'a', these are stored in file.

    def cropped_plot(self, filename=None, modelname=None, band=0, a_i=0, key=None):
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
        wl = np.array(w)*1e9
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
    
    def full_plot(self, filename=None, modelname=None, a_i=0, key=None):
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
            if max(th) > max_th: max_th = max(th) #unused I think
        title = r"Saturated Cherenkov Angle against Wavelength"
        if modelname is None: modelname = self.path
        title += "\n (" + modelname + ")"
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
            fig.savefig(filename, bbox_inches='tight')
        else:
            fig.show()

    def a_plot(self, filename=None, modelname=None):
        """Plot 'a' against Cherenkov angle and chromatic error
        Params:
        filename (str): Filename of exported graph
        modelname (str): String appended to title to distinguish graphs of different models
        """
        wl1, wl2, angle, err, a = self.a_matrix.T
        # exclude different wavelength ranges?
        #print(angle, err, a)
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(111)
        #ax.plot(a, angle)
        ax.errorbar(x=a, y=angle, yerr=err/2, color='black', capsize=5, marker='o', markersize=5) # error divided by 2 for errorbar
        title = r"Saturated Cherenkov Angle Against Unit Cell Size" # + str(np.round(wl1*1e9,3)) + r"-"+ str(np.round(wl2*1e9,3)
        if modelname is None: modelname = self.path
        title += " (" + modelname + ")"

        ax.set_title(title)
        ax.set_xlabel("Unit cell dimension")
        ax.set_xlabel(r"Unit cell dimension $a$ (m)")
        print(np.shape(angle))
        print(np.shape(err))

        ax.set_ylabel(r"Average Saturated Cherenkov Angle $\theta_c$ (rad)")
        print("Saving as", filename)
        fig.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
        else:
            fig.show()



