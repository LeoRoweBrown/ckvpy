import numpy as np 
import csv
import os
from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from . import bases
from .tools import effective

class Dispersion3D(bases.Bzone2D):
    """Intersect 2D band structure with electron plane. Can be used for 3D 
    data or 2D data, but 3D data must be confined to a 2D plane which at
    the moment is constrained to just planes in k_rho,kz where 
    k_rho = norm(kx,ky). Realistically in COMSOl this means only kz,kx and
    kz,ky because of the way bands are found - choose a direction from |k|=0
    and increase |k|. Always sorted by band (band is the root key)""" 
    def __init__(self, datafile, symmetry=4, headers=['skip', 'band', 'skip',
                'frequency', 'kx', 'ky', 'kz', 'n', 'skip'], sort_by = 'band',
                ndim=3, interpolate=True):

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

    def intersect(self, v=0.999, direction = [1,0]):
        """course intersection, then fine by looking in neighbourhood. 
        Do this by interpolating between lines?
        Electron direciton parameterised by direction
        :param
        v (float): electron speed
        direction (list): determines direction of electron with idices
        rho (|x,y|) and z which results in e-plane omega = k.v 
        """
        # raise NotImplementedError
        for band in self.data:
            m_rho = self.data[band]['mi']
            mz = np.copy(self.data[band]['mz'])  # mutated so copy
            my = self.data[band]['my']
            mx = self.data[band]['mx']
            mf = np.copy(self.data[band]['mf'])  # mutated so copy

            z_array = mz.T[0][-1:1:-1]  # starts at maximum
            rho_array = m_rho[0][1:-1] # cut off edges (interp)

            # e_plane = self.data[band]['mj']*3.e8*v
            mf *= 2*np.pi  # omega=2pif
            mf = mf.T # transposed z?
            self.data[band]['crossings'] = \
                {'ke': None, 'ko': None, 'f': None, 'direction': direction}
        # we are working to intersect in z direction, so loop over mj
        # transpose mj so we are row major (does this change actual data?)
        # order = 'F' (fortran order)
        # TODO redesign this, too confusing
            kz_c = np.array([])
            k_rho_c = np.array([])
            f_c = np.array([])
            for kz_i, kz in enumerate(z_array[:-1]):
                for k_rho_i, k_rho in enumerate(rho_array[:-1]):
                    kz2 = z_array[kz_i + 1]
                    k_rho2 = rho_array[k_rho_i + 1]
                    f = mf[kz_i, k_rho_i]
                    fz2 = mf[kz_i + 1, k_rho_i]
                    f_rho2 = mf[kz_i, k_rho_i + 1]
                    rho_found, rho_cross, z_found, z_cross = \
                        self._cross(v, kz, kz2, k_rho, k_rho2, f, fz2,
                                    f_rho2, direction)
                    k_rho_cross, f_rho_cross = rho_cross
                    kz_cross, fz_cross = z_cross
                    # rho_found = False # REMOVE TESTING
                    if z_found:  # no crossing found
                        kz_c = np.append(kz_c, kz_cross)
                        k_rho_c = np.append(k_rho_c, k_rho)
                        f_c = np.append(f_c, fz_cross)
                    if rho_found:
                        kz_c = np.append(kz_c, kz)
                        k_rho_c = np.append(k_rho_c, k_rho_cross)
                        f_c = np.append(f_c, f_rho_cross)
            # print("max",np.max(kz_c))
            self.data[band]['crossings']['kz'] = kz_c
            self.data[band]['crossings']['k_rho'] = k_rho_c
            self.data[band]['crossings']['f'] = f_c

        self.plotCherenkov()
            
    def _cross(self, v, kz, kz2, k_rho, k_rho2, f, fz2,
               f_rho2, direction):
        """Find crossings between electron plane in omega=v_z*kz+v_rho*k_rho
        and dispersion to find Cherenkov modes Interpolates between kz, kz2
        and k_rho, k_rho2 separately to find them.
        
        Params: 
        v (float): speed as percentage of c
        direction (list): from 0 to 1, component in rho and z direciton 
            respectively in form of [float, float]
        kz, kz2, k_rho, k_rho2 (float): ith and i+1th value of kz/k_rho in 
            ascending order
        f (float): value of frequency at kz k_rho
        fz2, f_rho2 (float): value of frequency at (k_rho,kz2) and (k_rho2,kz)

        returns:
        z_found, rho_found (bool): True if crossing found looking in the 
            kz/k_rho direction
        (kz_cross, fz_cross) tuple(float): kz and fz where crossing found
        (k_rho_cross, f_rho_cross) tuple(float) same for k_rho and f_rho
        """

        # first look along plane ke:
        ve = v*3.e8  # speed of light
        m_z = (fz2-f)/(kz2-kz)  # gradient
        m_rho = (f_rho2-f)/(k_rho2-k_rho)
        v_rho, v_z = ve*direction[0], ve*direction[1]
        v_abs = (v_rho**2 + v_z**2)**0.5
        # if abs(m_rho) < 1e-20 and abs(m_z) < 1e-20:
        #     kz_cross, k_rho_cross = f/v_z, f/v_rho 
        #     fz_cross = f
        #     f_rho_cross = f
        # else:

        cz = f - m_z*kz  # f intercept constant in f=m*k+c
        c_rho = f - m_rho*k_rho

        if abs(m_z - v_z) < 1e-15*abs(v_rho*k_rho - cz):
            z_found = False
            kz_cross = fz_cross = None
        else:
            kz_cross = (v_rho*k_rho - cz)/(m_z - v_z)
            fz_cross = kz_cross*m_z + cz
            z_found = True
        z = (kz_cross, fz_cross)

        if abs(m_rho - v_rho) < 1e-20*abs(v_z*kz - c_rho):
            rho_found = False
            k_rho_cross, f_rho_cross = None, None
        else:
            k_rho_cross = (v_z*kz - c_rho)/(m_rho - v_rho)
            f_rho_cross = k_rho_cross*m_rho + c_rho
            rho_found = True
        rho = (k_rho_cross, f_rho_cross)

        if rho_found:
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

    def plotCherenkov(self):
        fig = plt.figure(figsize=(10,8))
        fig1 = plt.figure(figsize=(10,8))

        dr, dz = self.data['1']['crossings']['direction']
        kz_axis = r"$k_z \,(m^{-1})$"
        k_rho_axis = r"$k_{\rho} \,(m^{-1})$"
 
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel(r" Cherenkov Angle $\theta_c$ (rad)")
        ax.set_ylabel(r"Wavelength $\lambda$ (m)")
        ax.set_ylim([0,1000])
        ax.set_title("Wavelength against Cherenkov Angle (+ve) \n Derived from 2D Dispersion")
        # ax_neg = fig.add_subplot(2,1,2)
        # ax_neg.set_xlabel(r" Cherenkov Angle $\theta_c$")
        # ax_neg.set_ylabel(r"Wavelength $\lambda$")
        # ax_neg.set_title("Wavelength against Cherenkov Angle (-ve) \n Derived from 2D Dispersion")
        # ax_neg.set_ylim([0,1000])
        # plt.subplots_adjust(vspace=0.4)
        fig.tight_layout()
        ax1 = fig1.add_subplot(1,1,1, projection='3d')
        ax1.set_title("Intersection Points Between Electron Plane \n and 2D Dispersion")
        ax1.set_xlabel(k_rho_axis)
        ax1.set_ylabel(kz_axis)
        ax1.set_zlabel(r"Frequency (Hz)")

        for band in self.data:
            print("Band", band + ":")
            kz = self.data[band]['crossings']['kz']
            k_rho = self.data[band]['crossings']['k_rho']
            f = self.data[band]['crossings']['f']
            adj_for_e_diretion = np.arctan(dz/(dr+1e-20))
            print(adj_for_e_diretion)
            theta = np.arctan(kz/(k_rho+1e-20)) - adj_for_e_diretion
            wl = 2*np.pi*3.e8/f*1e9

            pos_th, pos_wl, mean, err = \
                self._calc_err(theta, wl)
            neg_th, neg_wl, neg_mean, neg_err = \
                self._calc_err(theta, wl, sign=-1)

            ax.plot(theta, wl, linestyle='None', marker='o')
            ax.set_xticks(np.arange(0,0.4+0.004, 0.05))
            ax.set_xlim([0, max(pos_th)+0.05])
            # ax_neg.set_xticks(np.arange(0,-0.4-0.05, -0.05))
            # ax_neg.set_xlim([min(pos_th), max(pos_th)+0.05])
            # ax_neg.plot(neg_th, neg_wl, linestyle='None', marker='o')
            global_max = max([np.max(kz), np.max(k_rho)])
            ax1.set_ylim([-global_max, global_max])
            ax1.set_xlim([-global_max, global_max])
            ax1.invert_xaxis()
            ax1.scatter(k_rho, kz, f) # reversing for viewing ease
            # self.data[band]['crossings']['pos_theta'] = pos_th
            self.data[band]['crossings']['theta'] = theta
            self.data[band]['crossings']['wavelength'] = wl
            self.data[band]['crossings']['error'] = err

        fig.savefig("cherenkov.png", bbox_inches='tight')
        fig1.savefig("intersection.png", bbox_inches='tight')
        fig.show()
        fig1.show()
    
    def _calc_err(self, theta, wl, sign=1):
        wl_2_5nm = []
        theta_2_5nm = []
        mean = None
        err = None
        for i, w in enumerate(wl):
            if w < 1000 and w > 250 and sign*theta[i]>0:
                wl_2_5nm.append(w)
                theta_2_5nm.append(theta[i])
        try:
            mean = np.mean(theta_2_5nm)
            err = abs(np.max(theta_2_5nm)-np.min(theta_2_5nm))
            print("Angle +ve", mean)
            print("Chromatic error +ve", err)
        except ValueError:
            print("None found")
        return theta_2_5nm, wl_2_5nm, mean, err

    def plot3D(self, mode='surface'):
        print("Reflecting")
        self.reflect()
        print("Interpolating")
        self.interpolate()
        print("Plotting")
        fig = plt.figure(figsize=(12,9))

        for i, band in enumerate(self.data):
            mf = self.data[band]['mf']
            m_rho = self.data[band]['mi']
            mz = self.data[band]['mj']

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
                # ax.scatter(self.data['1']['k_rho'], self.data['1']['kz'], self.data['1']['frequency'])
                ax.scatter(m_rho, mz, mf)
            elif mode == 'eplane':
                plane = ax.plot_surface(m_rho, mz, mz*0.999*3.e8, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
               
            #ax.set_zlim([np.min(mf),np.max(mf)])
            fig.savefig("dispersion"+band+".png", bbox_inches='tight')

    def compare_medium(self, index="sio2"):
        print(__file__)
        index_file = os.path.join(os.path.dirname(__file__),\
            "index\\"+ index + ".txt")
        wl_sio2, n_sio2 = np.loadtxt(index_file).T
        for band in self.data:
            th = self.data[band]['crossings']['theta']
            wl = self.data[band]['crossings']['wavelength']*1e-9
            err = self.data[band]['crossings']['error']
            ind = np.argsort(wl)
            wl = np.array([wl[i] for i in ind])
            th = np.array([th[i] for i in ind])
            n_sio2_interp = np.interp(wl, wl_sio2, n_sio2)
            e_sio2 = n_sio2_interp*n_sio2_interp
            # volume ratio 2d
            v_r = (np.pi*(0.45*(3**0.5))**2)/(3*3**0.5/2)
            # volume ratio z direction
            v_rz = 2./3.
            eff_2d = effective.index(e_sio2, 1.0, v_r) # goes wrong
            eff = effective.index(eff_2d, 1.0, v_rz)
            n_eff = eff**0.5
            n_data = 1./np.cos(th)
            th_eff = np.arccos(1./n_eff)
            n_err_p = abs(1./np.cos(th + np.ones_like(th)*err) - n_data)
            n_err_n = abs(1./np.cos(th - np.ones_like(th)*err) - n_data)
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            ax.plot(wl*1e9, n_eff, label="Effective medium theory", \
                color='black', linestyle='-', marker='*', markersize=5)
            ax.plot(wl*1e9, n_data,
                label="Simulation", color='black', marker='o',\
                markersize=5)
            ax.set_xlim([np.min(wl),1000])
            ax.set_ylim([np.min(n_eff)-0.005, np.max(n_data)+0.005])
            ax.set_title("Effective Index Comparison Between Theory and Simulation")
            ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
            ax.set_ylabel(r"Refractive index $n_{eff}$")
            ax.legend()
            ax.axvline(250, linestyle='dashed', color='black')
            ax.axvline(500, linestyle='dashed', color='black')
            ax1 = ax.twinx() # fig.add_subplot(212)
            # ax1.plot(wl*1e9, th_eff, label="Effective medium theory", \
            #     color='black', linestyle='-', marker='*', markersize=5)
            # ax1.plot(wl*1e9, np.abs(th),
            #     label="Simulation", color='black', marker='o',\
            #     markersize=5)
            ax1.set_xlim([np.min(wl),1000])

            ax1.set_ylim([np.arccos(1./(np.min(n_eff)-0.005)), np.arccos(1./(np.max(n_data)+0.005))])
            ax1.set_xlabel(r"Wavelength $\lambda$ (nm)")
            ax1.set_ylabel(r"Saturated Cherenkov Angle $\theta_c$ (rad)")
            # ax1.legend()
            fig.savefig("effective_index.png")
            fig.show()

    