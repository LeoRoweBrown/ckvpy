import numpy as np 
import csv
from os import listdir
from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from . import bases

class Dispersion3D(bases.Bzone2D):
    """Intersect 2D band structure with electron plane. Can be used for 3D 
    data or 2D data, but 3D data must be confined to a 2D plane which at
    the moment is constrained to just planes in k_rho,kz where 
    k_rho = norm(kx,ky). Realistically in COMSOl this means only kz,kx and
    kz,ky because of the way bands are found - choose a direction from |k|=0
    and increase |k|. Always sorted by band (band is the root key)""" 
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

    def intersect(self, v=0.999, direction = [0,0,1]):
        """course intersection, then fine by looking in neighbourhood. 
        Do this by interpolating between lines?
        Electron direciton parameterised by direction"""
        # raise NotImplementedError
        for band in self.data:
            mi = self.data[band]['mi']
            mz = np.copy(self.data[band]['mz'])  # mutated so copy
            my = self.data[band]['my']
            mx = self.data[band]['mx']
            mf = np.copy(self.data[band]['mf'])  # mutated so copy

            z_array = mz.T[0][-1:1:-1]  # starts at maximum
            rho_array = mi[0][1:-1] # cut off edges
            y_array = my[0][1:-1]
            x_array = mx[0][1:-1]
            ke = np.zeros_like(rho_array)
            k_out = np.zeros_like(rho_array)
            # generalised, but usually direction is only [1,0] or [0,1]
            if (direction[0] == 0 and direction[1] == 0) and direction[2] != 0:
                print("ke is kz")
                ke_a = z_array
                k_out_plane = rho_array
            elif direction[2] == 0:
                print("ke is krho")
                ke_a = rho_array
                k_out_plane = z_array
                mf = mf.T  # because we access f = mf[ko_i, ke_i] (swap indices)
                # and mf is always mf[k_z, k_rho]
            else:
                raise ValueError("Direction must be in direction of rho or z "
                                "only x, y and z implemented at the moment")

            # e_plane = self.data[band]['mj']*3.e8*v
            mf *= 2*np.pi  # omega=2pif
            self.data[band]['crossings'] = \
                {'ke': None, 'ko': None, 'f': None, 'direction': direction}
        # we are working to intersect in z direction, so loop over mj
        # transpose mj so we are row major (does this change actual data?)
        # order = 'F' (fortran order)

            ke_c = k_out_c = f_c = np.array([])
            for ke_i, ke in enumerate(ke_a[:-1]):
                for ko_i, ko in enumerate(k_out_plane[:-1]):
                    ko2 = k_out_plane[ko_i + 1]
                    ke2 = ke_a[ke_i + 1]
                    f = mf[ko_i, ke_i]
                    f2e = mf[ko_i, ke_i + 1]
                    f2o = mf[ko_i + 1, ke_i]

                    fe_cross, ke_cross, found_e = \
                        self._cross(v, ke, ke, ke2, f, f2e, True)
                    fo_cross, ko_cross, found_o = \
                        self._cross(v, ke, ko, ko2, f, f2o, False)
                    if found_o or found_e:
                        # print("found crossing")
                        if found_o:
                            f_cross = fe_cross
                        elif found_e:
                            f_cross = fo_cross
                        else:
                            f_cross = (fe_cross + fo_cross)/2.
                    else:  # no crossing found
                        f_cross = None
                        continue
                    # temp arrays for readability
                    ke_c = np.append(ke_c, ke_cross)
                    k_out_c = np.append(k_out_c, ko_cross)
                    f_c = np.append(f_c, f_cross)
            self.data[band]['crossings']['ke'] = ke_c
            self.data[band]['crossings']['ko'] = k_out_c
            self.data[band]['crossings']['f'] = f_c
        self.plotCherenkov()
            
    def _cross(self, v, ke, k, k2, f, f2, inplane):

        # first look along plane ke:
        ve = v*3.e8  # speed of light
        m = (f2-f)/(k2-k)  # gradient
        if abs(m) < 1e-20:
            k_cross = f/ve  # in which k direction? ambiguous atm but works
            f_cross = f
        else:
            c = f - m*k
            if inplane:
                k_cross = c/(ve - m)
            else:  # for out of plane case where k=/=ke
                k_cross = (ke*ve - c)/m
            f_cross = k_cross*m + c
        if k_cross <= k2 and k_cross >= k and f_cross >= f \
            and f_cross <= f2 and f_cross > 1.e-15:
            # print("found crossing in _find_cross")
            return f_cross, k_cross, True
        else:
            return f, ke, False
            # temp arrays for readability


    def plotCherenkov(self):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(2,1,1)
        ax.set_xlabel(r" Cherenkov Angle $\theta_c$")
        ax.set_ylabel(r"Wavelength $\lambda$")
        # self.data[band]['crossing']['kz'] = 
        #theta = np.arctan(k_out_c/(ke_c+1e-20))
        ax.set_ylim([0,1000])
        ax.set_title("Wavelength against Cherenkov Angle \n Derived from 2D Dispersion")
        direction = self.data['1']['crossings']['direction']
        if direction[2] != 0:
            ke_axis = r"$k_z$"
            ko_axis = r"$k_{\rho}$"
        else:
            ke_axis = r"$k_{\rho}$"
            ko_axis = r"$k_z$"
        ax1 = fig.add_subplot(2,1,2, projection='3d')
        
        ax1.set_xlabel(ke_axis)
        ax1.set_ylabel(ko_axis)
        for band in self.data:
            ke = self.data[band]['crossings']['ke']
            ko = self.data[band]['crossings']['ko']
            f = self.data[band]['crossings']['f']
            theta = np.arctan(ke/(ko+1e-20))
            wl = 2*np.pi*3.e8/f*1e9
            wl_2_5nm = []
            theta_2_5nm = []
            for i, w in enumerate(wl):
                if w < 1000 and w > 250 and theta[i]>0:
                    wl_2_5nm.append(w)
                    theta_2_5nm.append(theta[i])
            print("band", band, "Angle", np.mean(theta_2_5nm))
            print("band", band, "Chromatic error", 
                  abs(np.max(theta_2_5nm)-np.min(theta_2_5nm)))

            ax.plot(theta, wl, linestyle='None', marker='o')
            ax.set_xticks(np.arange(0,0.4+0.05, 0.05))
            ax.set_xlim([0, 0.4+0.05])
            ax1.scatter(ke, ko, f)
        fig.show()

    def plot3D(self, mode='surface'):
        print("Reflecting")
        self.reflect()
        print("Interpolating")
        self.interpolate()
        print("Plotting")
        fig = plt.figure(figsize=(12,9))
        #ax = fig.gca(projection='3d')

        for i, band in enumerate(self.data):
            mf = self.data[band]['mf']
            mi = self.data[band]['mi']
            mj = self.data[band]['mj']

            #ax = fig.add_subplot(2,3,i+1, projection='3d')
            ax = fig.add_subplot(1,1,1, projection='3d')
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
            print('rho', np.max(mi), 'z', np.max(mj))
            if mode == 'surface':
                surf = ax.plot_surface(mi, mj, mf, cmap=cm.bwr,
                                       linewidth=0, antialiased=False)
                #surf = ax.plot_surface(mi, mj, mi*3.e8, cmap=cm.bwr,
                #                       linewidth=0, antialiased=False)
            elif mode == 'scatter':
                ax.scatter(self.data['1']['k_rho'], self.data['1']['kz'], self.data['1']['frequency'])
            # plane = ax.plot_surface(mi, mj, mj*0.9*3.e8*np.pi/a, cmap=cm.coolwarm,
            #                 linewidth=0, antialiased=False)
               
            #ax.set_zlim([np.min(mf),np.max(mf)])
        fig.show()
    