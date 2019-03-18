import numpy as np 
import csv
from os import listdir
from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from . import bases

class Dispersion3D(bases.Bzone2D):
    """Intersect 2D band structure with electron plane. Can be used for 3D data or 2D data, but 3D data must be confined to a 2D
    plane which at the moment is constrained to just planes in k_rho,kz where k_rho = norm(kx,ky). Realistically in COMSOl this
    means only kz,kx and kz,ky because of the way bands are found - choose a direction from |k|=0 and increase |k|. Always sorted
    by band (band is the root key)""" 
    def __init__(self, datafile, symmetry, headers=['skip', 'band', 'skip', 'frequency',\
        'kx', 'ky', 'kz', 'n', 'skip'], sort_by = 'band', ndim=3, interpolate=True):

        plt.rcParams.update({'font.size': 14})
        np.set_printoptions(precision=3) # will this affect data saved to text?
        self.ndim = ndim
        for header in headers:
            if header not in \
            ['band', 'frequency', 'kx', 'ky', 'kz', 'n', 'skip']:
                print(header, "not allowed.")
                raise ValueError("Invalid header supplied, must be one of "
                    "['band', 'frequency', 'kx', 'ky', 'kz', 'n']")
        super(Dispersion3D, self).__init__\
            (datafile, headers=headers, ndim=ndim,\
            symmetry=symmetry)
        if interpolate:
            self.reflect()
            self.interpolate()  # default resolution is 100 elements

    def intersect(self, v=0.999):
        """course intersection, then fine by looking in neighbourhood. Do this by interpolating between lines?"""
        # raise NotImplementedError
        for band in self.data:
            # e_plane = self.data[band]['mj']*3.e8*v
            mf = self.data[band]['mf']  # is this by reference?
            mj = self.data[band]['mj']
            mi = self.data[band]['mi']
            self.data[band]['crossings'] = {'kz': [], 'k_rho': [], 'f': []}
        # we are working to intersect in z direction, so loop over mj
        # transpose mj so we are row major (does this change actual data?)
        # order = 'F' (fortran order)
            # get array for z values and rho values
            z_array = mj.T[0][:-1]  # starts at maximum
            rho_array = mi[0]

            for rho_i, z_values in enumerate(rho_array):
                #print(x, xi)
                for z_i, z in enumerate(z_array[:-1]):
                    #print(y, yi)
                    z2 = z_array[z_i + 1]
                    f = mf[rho_i, z_i]
                    f2 = mf[rho_i, z_i + 1]
                    ve = v*3.e8  # speed of light
                    # print(z2)
                    # print(z)
                    m = (f2-f)/(z2-z)  # gradient
                    rho = rho_array[rho_i]
                    #y_cross = y*ve # ve*k = f
                    # print(m)
                    if abs(m) < 1e-20:
                        #print('skipping')
                        z_cross = f/ve
                    else:
                        c = f - m*z
                        z_cross = c/(ve - m)
                    # check if crossing point is inside ROI (f2-f, z2-z, rho)
                    if abs(z_cross) <= np.max(z_array) and z_cross*ve >= f and z_cross*ve <= f2:
                        #print(y)
                    #    print('crossing at y=', y_cross,'x=',x,'f=',y_cross*ve)
                        self.data[band]['crossings']['kz'].append(z) # replace with intialised lists for speed?
                        self.data[band]['crossings']['k_rho'].append(rho)
                        self.data[band]['crossings']['f'].append(f)

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel(r"$k_rho$")
            ax.set_ylabel(r"Wavelength $\lambda$")
            # zmax = np.max(z_array)
            # rhomax = np.max(rho_array)
            theta = np.arctan(rho_array/(z_array+1e-20))  # <-- better sol
            wl = 2*np.pi/self.data[band]['crossings']['f']
            ax.set_title("Wavelength against Cherenkov Angle Derived from 2D Dispersion")
            ax.plot(theta, wl)

            plt.show()
    
    def plot3D(self, mode='surface'):
        print("Reflecting")
        self.reflect()
        print("Interpolating")
        self.interpolate(resolution=0.01)
        print("Plotting")
        fig = plt.figure(figsize=(12,9))
        #ax = fig.gca(projection='3d')
        for i, band in enumerate(self.data):
            mf = self.data[band]['mf']
            mi = self.data[band]['mi']
            mj = self.data[band]['mj']

            #ax = fig.add_subplot(2,3,i+1, projection='3d')
            ax = fig.add_subplot(1,1,1, projection='3d')
            mi_range = np.max(mi) - np.min(mi)
            mj_range = np.max(mj) - np.min(mj)
            global_max = np.max([np.max(mi), np.max(mj)])
            global_min = np.min([np.min(mi), np.min(mj)])
            ax.set_xlim([global_min, global_max])
            ax.set_ylim([global_min, global_max])
            ax.auto_scale_xyz([np.min(mi), np.max(mi)], [np.min(mj), np.max(mj)], [np.min(mf),np.max(mf)])
            # print(mj_range/mi_range)
            #ax.set_aspect(mj_range/mi_range)
            ax.set_title(r"Band "+str(i+1)+r" 3D Model Dispersion")
            ax.set_xlabel(r"Wavevector in $(k_x,k_y)$, $k_{\rho}$ $(m^{-1})$")
            ax.set_ylabel(r"Wavevector $k_z$ $(m^{-1})$")
            ax.set_zlabel(r"Frequency $(Hz)$")
            if mode == 'surface':
                surf = ax.plot_surface(mi, mj, mf, cmap=cm.bwr,
                                       linewidth=0, antialiased=False)
            elif mode == 'scatter':
                ax.scatter(self.data['1']['k_rho'], self.data['1']['kz'], self.data['1']['frequency'])
            # plane = ax.plot_surface(mi, mj, mj*0.9*3.e8*np.pi/a, cmap=cm.coolwarm,
            #                 linewidth=0, antialiased=False)
               
            #ax.set_zlim([np.min(mf),np.max(mf)])
        fig.show()
    