import math
import numpy as np
from scipy import interpolate
from ckvpy.tools.csvloader import CSVLoader

class Bzone2D(CSVLoader):
    """Gets all values of kx,ky,(kz) and f from dispersion data and creates
    full Brillouin zone to make a 2D dispersion. Uses interpolation. 
    Implementation for 2D model/crystal exists, but probably has limited 
    use-case"""
    
    def __init__(self, datafile, symmetry, 
                headers=['skip', 'band', 'skip' 'frequency',\
                'kx', 'ky', 'kz', 'n', 'skip'], sort_by = 'band', ndim=3):
        
        print("Using Bzone2D")

        np.set_printoptions(precision=3) # will this affect data saved to text?
        self.ndim = ndim
        self.symmetry = symmetry
        self.reflected = False
        self.status = {'reflected': False, 'interpolated':False, \
            'intersected': False}

        for header in headers:
            if header not in \
            ['band', 'frequency', 'kx', 'ky', 'kz', 'n', 'skip']:
                raise ValueError("Invalid header supplied, must be one of"
                    "['band', 'frequency', 'kx', 'ky', 'kz', 'n']")
        return super(Bzone2D, self).__init__\
            (datafile, headers=headers, sort_by = sort_by)

    def _convert_to_polar(self, inverse=False):
        for band in self.data:
            kx = np.array(self.data[band]['kx'])
            ky = np.array(self.data[band]['ky'])
            kz = np.zeros_like(ky)
            k_theta = np.zeros_like(ky)  # in plane angle
            k_rho = np.sqrt(kx*kx + ky*ky)  # in plane component

            if self.ndim == 2:
                k_abs = k_rho
                for i in range(len(kz)):
                    if k_abs < 1e-10:
                        k_theta[i] = 0.0
                    elif kx[i] < 1e-10 and k_abs[i] > 1e-15:
                        k_theta[i] = np.pi/2.
                    elif kx[i] > 1e-10:
                        k_theta[i] = np.arctan(ky[i]/kx[i])
                k_phi = np.zeros_like(ky)

            if self.ndim == 3:
                kz = np.array(self.data[band]['kz'])
                # k_theta = np.arctan((kx*kx + ky*ky)**0.5/kz)
                k_abs = np.sqrt(kx*kx + ky*ky + kz*kz)
                k_phi = np.zeros_like(kz)
                
                for i in range(len(kx)): # check for kx = 0 case
                    if kx[i] < 1e-10 and k_rho[i] > 1e-10: 
                        print("angle phi is 90 degrees")
                        k_phi[i] = np.pi/2.
                    elif kx[i] < 1e-10 and k_rho[i] < 1e-10:
                        # value doesnt matter, k_rho*anything = 0
                        k_phi[i] = 0.0  
                    # angle of plane our data lies on to kx axis in 3D space
                    elif kx[i] > 1e-15:
                        k_phi[i] = np.arctan(ky[i]/kx[i])
                    
                    if k_abs[i] < 1e-10:
                        k_theta[i] = 0.0
                    elif kz[i] < 1e-10 and k_abs[i] > 1e-10:
                        k_theta[i] = np.pi/2.
                    elif kz[i] > 1e-10:
                        k_theta[i] = \
                            np.arctan((kx[i]*kx[i] + ky[i]*ky[i])**0.5/kz[i])
                # standard deviation ignoring angles for k_rho<1e-10 since
                # these were set to 0.0 and could be wrong
                if np.std(k_phi[k_rho<1e-10]) > 1e-10*np.mean(k_phi):
                    raise ValueError(
                            "Data must be in a form where the BZ "
                            "plane is defined by xz or yz (or (x,y)z with "
                            "y/x fixed - angle between kx and ky must be "
                            "constant")

            # print(type(k_abs), type(k_phi), type(k_theta), type(k_rho))
           
            self.data[band]['k_abs'] = k_abs
            self.data[band]['k_theta'] = k_theta
            self.data[band]['k_rho'] = k_rho
            self.data[band]['k_phi'] = k_phi
            lengths = [len(k_abs), len(k_theta), len(k_rho), len(k_phi)]
            if lengths.count(lengths[0]) != len(lengths):
                raise IndexError("Data arrays do not have"
                                 "same lengths!")

    def reflect(self, symmetry=4):
        """
        Use symmetry of irreducible Brillouin zone to reproduce full Brillouin
        zone. For 3D model, only valid for kx=0 or ky = 0 planes.
    
        Params:
        symmetry (int): Degree of symmetry (of Dihedral group D_n, n=symmetry)
        start_angle (int) Angle (DEGREES) in BZ that data starts at in
            degrees, assumed to be minimum angle
        """
        if self.status['reflected']:
            print("Already reflected!")
            return
        # transformations of angle, first number is multiplier, second is addition
        # reflect in theta = 0 
        reflect_th_0 = [-1.0, 0.0] 
        reflect_th_min45 = [-1.0, -90.0*np.pi/180.0] 
        reflect_th_45 = [-1.0, 90.0*np.pi/180.0]
        reflect_th_90 = [-1.0, np.pi] 
        # but sin(180+theta) = sin(-180+theta)

        if symmetry == 4:
            reflections = [reflect_th_0, reflect_th_90]
        elif symmetry == 8:
            reflections = [reflect_th_0, reflect_th_min45, reflect_th_45]
        elif symmetry == 6 or symmetry == 12:
            raise NotImplementedError(
                "Symmetry of order multiple of 6 not yet implemented"
                "e.g. for the hexagonal unit cell/reciprocal lattice"
                )
        else:
            raise ValueError("Please pick symmetry order 4 or 8")

        self._convert_to_polar()

        for band in self.data:
            start_angle = np.min(self.data[band]['k_theta']) 
            if abs(start_angle) > 1e-10:
                raise ValueError("Please only use data that starts"
                    "from 0 degrees")
            #print(band)
            for params in reflections:
                new_angles = params[0]*np.array(self.data[band]['k_theta']) \
                    + params[1]
                #plane_angle = np.average(self.data[band]['plane_angle'])
                # tile copies array and appends
                self.data[band]['k_abs'] = \
                    np.tile(self.data[band]['k_abs'],2) 
                self.data[band]['k_theta'] = \
                    np.concatenate((self.data[band]['k_theta'], new_angles))
                self.data[band]['frequency'] = \
                    np.tile(self.data[band]['frequency'], 2)
                self.data[band]['k_rho'] = \
                    np.tile(self.data[band]['k_rho'], 2)
                self.data[band]['k_phi'] = \
                    np.tile(self.data[band]['k_phi'], 2)

            # now use full k_abs and k_theta data to derive kx, ky kz
            if self.ndim == 3:
                k_rho = np.array(self.data[band]['k_abs'])*\
                    np.sin(self.data[band]['k_theta'])
                self.data[band]['k_rho'] = k_rho
                self.data[band]['kx'] = k_rho*np.cos(self.data[band]['k_phi'])
                self.data[band]['ky'] = k_rho*np.sin(self.data[band]['k_phi'])
                self.data[band]['kz'] = np.array(self.data[band]['k_abs'])*\
                    np.cos(self.data[band]['k_theta'])
            elif self.ndim == 2:
                self.data[band]['kx'] = self.band[band]['k_abs']*\
                    np.cos(self.data[band]['k_theta'])
                self.data[band]['ky'] = self.band[band]['k_abs']*\
                    np.sin(self.data[band]['k_theta'])
        self.status['reflected'] = True

    def interpolate(self, resolution=100, method='cubic'):
        """ work directly in k space with k values (not fraction of brillouin
        zone) """
        if not self.status['reflected']:
            self.reflect(self.symmetry)

        for band in self.data:
            if self.ndim == 3:
                ki_data = self.data[band]['k_rho']
                kj_data = self.data[band]['kz']

            elif self.ndim == 2:
                ki_data = self.data[band]['kx']
                kj_data = self.data[band]['ky']

            f_data = self.data[band]['frequency']

            print("Hard zeroes in data?", 0 in f_data)
            print("NaNs in data?", math.nan in f_data)
            ki_max = np.max(ki_data)
            ki_min = np.min(ki_data)
            kj_max = np.max(kj_data)
            kj_min = np.min(kj_data)
            
            noelements_i = int(resolution)#*ki_max/max((ki_max, kj_max))
            noelements_j = int(resolution)#*kj_max/max((ki_max, kj_max))
            ki = np.linspace(ki_min, ki_max, noelements_i)
            kj = np.linspace(kj_min, kj_max, noelements_j)
            mi, mj = np.meshgrid(ki,kj)
            mj = np.copy(mj[::-1]) # reverse so that lowest kj is at bottom
            mf = np.zeros_like(mi) # 

            # outside of convex hull the values are NaN, so replace these with
            # nearest neighbour
            interpolate_hull = interpolate.griddata((ki_data,\
                kj_data), f_data, (mi,mj), method=method)
            fill_matrix = interpolate.griddata((ki_data,\
                kj_data), f_data, (mi,mj), method='nearest')
                        
            interpolate_hull[np.isnan(interpolate_hull)] = 0
            mask = np.ones_like(interpolate_hull) * (interpolate_hull==0)
            # invert mask with mask==0
            mf = mask*fill_matrix + interpolate_hull*(mask==0) 
            mf[np.isnan(mf)] = 0
            #print(mf)
            if np.nan in mf:
                raise Warning("NaNs in matrix")

            phi = np.mean(self.data[band]['k_phi'])
            self.data[band]['mx'] = \
                mi*np.cos(phi)
            self.data[band]['my'] = \
                mi*np.sin(phi)
            self.data[band]['mz'] = mj
            self.data[band]['mf'] = mf
            self.data[band]['mi'] = mi
            self.data[band]['mj'] = mj
        self.status['interpolated'] = True

    def addzero(self, band):
        """Add point f(k=0) = 0 to help interpolation"""
        self.data[band]['frequency'].append(0.0)
        self.data[band]['kx'].append(0.0)
        self.data[band]['ky'].append(0.0)
        self.data[band]['kz'].append(0.0)

    def _removerawnans(self):
        """Attempt to fix interpolation with missing data by
        removing data points with NaNs"""
        for band in self.data:
            for i, f in enumerate(self.data[band]['frequency']):
                if math.isnan(f):
                    for param in self.data[band]:
                        if len(self.data[band][param]) == \
                            len(self.data[band]['frequency']):
                                np.delete(self.data[band][param], i)