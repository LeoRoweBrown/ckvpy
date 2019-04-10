import numpy as np 
import scipy.constants as const

class dataAnalysis(object):
    """Class to handle wavelength cuts, sorting, angle finding etc."""
    def __init__(self, data):
        self.data_dict = data

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
        for a in self.data_dict:
            for band in self.data_dict[a]:
                print("Finding angle for a =", a, "band", band)
                wl1, wl2, average, rnge = self.calc_err(wl_range, a=a, band=band)
                array = np.array([wl1, wl2, average, rnge, float(a)])
                self.data_dict[a][band]['cherenkov'] = array.tolist()  # json friendly
                # self.data_dict[a][band][str(wl1)+'-']

        # print(average, rnge)
        if self.format is 'single':
            return average, rnge
        return # dont return average and range if computed for multiple values of 'a', these are stored in file.
    
    def n_eff(self, method='gradient'):
        """method is 'gradient' or 'angle', TODO: may remove"""
        for root in self.data_dict:
            for band in self.data_dict[root]:
                data = self.data_dict[root][band]
                if 'n' in data:
                    print('refractive index already in data')
                    continue
                if 'kx' in data and 'ky' in data and 'ky' in data:
                    kx = data['kx']
                    ky = data['ky']
                    kz = data['kz']
                    k0 = 2*np.pi*data['frequency']/const.c # omega/c
                    data['neff'] = sqrt(kx*kx+ky*ky+kz*kz)**2./k0

    def calc_err(self, wl_range, a='default', band='0', sign=1):
        """Find chromatic error and remove negative/positive angles between
        for wavelengths between 250nm and 500nm.
        Args:
            theta (array[float]): array of cherenkov angles found from 
                intersection
            wavelength (array[float]) array of wavelengths assoicated
                with theta array
            sign (int): whether to accept positive (1) or negative (-1) angles

        Returns:
            (list), (list), (float), (float): list of angles between 
            250-500nm, list of wavelengths for these angles, mean of
            angles and range of angles
        """
        print("Removing negative angles")
        self.sort_data('wavelength') 
        wl_pos, th_pos = self.wl_cut(a, band)  # take positive angles
        self.data_dict[a][band]['angle'] = th_pos
        self.data_dict[a][band]['wavelength'] = wl_pos
        print(wl_pos)

        wl_interp1, angle_interp1, i1 = \
            self._interp_angle(wl_range[0], a, band)
        wl_interp2, angle_interp2, i2 = \
            self._interp_angle(wl_range[1], a, band)
        
        try:
            mean = np.average(th_pos[i1:i2])
            err = abs(angle_interp2-angle_interp1)
            print("Angle", mean)
            print("Chromatic error", err)
        except ValueError:
            print("None found")
            mean = None
            err = None
        
        return wl_interp1, wl_interp2, mean, err

    def _interp_angle(self, wavelength, a='default', band='0'):
        raise NotImplementedError

        """Interpolate between two points of data to find angle at desired 
        wavelength for given band e.g. wavelength=250nm -> interpolate angle
        between between points either side of 250nm
        
        Params:
        wavelength (float): desired wavelength (in m or same unit as data) at
            which to solve for angle 
        a (str): key for unit cell size in data dictionary
        """
        i = 0
        band = str(band)
        # make sure in ascending order for line 
        # "while float(wl[i]) < wavelength"
        wl = self.data_dict[a][band]['wavelength']
        th = self.data_dict[a][band]['angle']
        
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

    def wl_cut(self, wl, wl_range=[0.,1e10], sign=1, param_key=None):
        """Take cut of data based on wavelength range. Default behaviour
        removes negative angles"""
        wl_nm_range = []
        theta_nm_range = []
        param_nm_range = []
        for i, w in enumerate(wl):
            if w < wl_range[1] and w > wl_range[0] and sign*theta[i]>0:
                wl_nm_range.append(w)
                theta_nm_range.append(theta[i])
                if param is not None:
                    param_nm_range.append(param[i])
        if param is None:
            return theta_nm_range, wl_nm_range
        else:
            return theta_nm_range, param_nm_range, wl_nm_range

class dataAnalysis3D(dataAnalysis):
    def __init__(self, data):
        self.data_full = data
        self.data_dict = {}
        self._init_data_dict(data)
        return super().__init__(data)
    
    def _init_data_dict(self, data):
        data_dict = {}
        for root in self.data_full:
            data_dict[root]
    
    def calculateCherenkov(self, beta=0.999, direction = [1,0],
                          wl_range=[250.e-9,500.e-9]):
        """course intersection, then fine by looking in neighbourhood. 
        Do this by interpolating between lines?
        Electron direciton parameterised by direction
        Args:
            beta (float): electron speed ratio with c
            direction (list): determines direction of electron with idices
                rho (|x,y|) and z which defines e-plane omega = k.v
            wl_range list[int/float, int/float]: range of wavelength in nm
                to analyse Cherenkov angle and chromatic error over
        """
        for band in self.data_full['default']:
            m_rho = self.data_full['default'][band]['mi']  # matrix of k_rho values
            mz = np.copy(self.data_dict['default'][band]['mz'])  # mutated so copy
            my = self.data_full['default'][band]['my']  # matrix of ky values
            mx = self.data_full['default'][band]['mx']  # matrix of kx values
            mf = np.copy(self.data_full['default'][band]['mf'])  # mutated so copy

            z_array = mz.T[0][-1:1:-1]  # starts at maximum
            rho_array = m_rho[0][1:-1]  # cut off edges (interp)

            # e_plane = self.data_dict['default'][band]['mj']*3.e8*v
            mf *= 2*np.pi  # omega=2pif
            mf = mf.T # since we transpose z to get z array from columns
            self.data_dict['default'][band] = \
                {'ke': [None], 'ko': [None], 'frequency': [None], 'direction': direction}

            kz_c = np.array([])  # empty temp arrays to store crossing points
            k_rho_c = np.array([])
            f_c = np.array([])
            for kz_i, kz in enumerate(z_array[:-1]):  # ith value of kz
                for k_rho_i, k_rho in enumerate(rho_array[:-1]):  # jth k_rho
                    kz2 = z_array[kz_i + 1]  # i+1th value of kz
                    k_rho2 = rho_array[k_rho_i + 1]  # j+1th k_rho
                    f = mf[kz_i, k_rho_i]  # f(kz,k_rho)
                    fz2 = mf[kz_i + 1, k_rho_i]  # f(kz2,k_rho)
                    f_rho2 = mf[kz_i, k_rho_i + 1]  # f(kz,k_rho2)
                    # get crossing points and booleans
                    rho_found, rho_cross, z_found, z_cross = \
                        self._cross(beta, kz, kz2, k_rho, k_rho2, f, fz2,
                                    f_rho2, direction)
                    k_rho_cross, f_rho_cross = rho_cross
                    kz_cross, fz_cross = z_cross
                    if z_found:  # crossing found in kz direction
                        kz_c = np.append(kz_c, kz_cross)
                        k_rho_c = np.append(k_rho_c, k_rho)
                        f_c = np.append(f_c, fz_cross)
                    if rho_found:  # crossing found in k_rho direction
                        kz_c = np.append(kz_c, kz)
                        k_rho_c = np.append(k_rho_c, k_rho_cross)
                        f_c = np.append(f_c, f_rho_cross)
            self.data_dict['default'][band]['kz'] = kz_c
            self.data_dict['default'][band]['k_rho'] = k_rho_c
            # set back to f instead of omega
            self.data_dict['default'][band]['frequency'] = f_c/(2*np.pi)
            if len(self.data_dict['default'][band]['kz']) == 0:
                raise Warning("No intersection found between electron plane "
                            "and dispersion plane,")
            self.status['intersected'] = True

    def _cross(self, beta, kz, kz2, k_rho, k_rho2, f, fz2,
               f_rho2, direction):
        """Find crossings between electron plane in omega=v_z*kz+v_rho*k_rho
        and dispersion to find Cherenkov modes Interpolates between kz, kz2
        and k_rho, k_rho2 separately to find them.
        
        Args: 
            beta (float): speed as percentage of c
            direction (list): from 0 to 1, component in rho and z direciton 
                respectively in form [float, float] ([1,1] -> velocity=(v,v))
            kz, kz2, k_rho, k_rho2 (float): ith and i+1th value of kz/k_rho
                in ascending order
            f (float): value of frequency at kz k_rho
            fz2, f_rho2 (float): value of frequency at (k_rho,kz2) and 
                (k_rho2,kz)

        Returns:
            z_found, rho_found (bool): True if crossing found looking in the 
                kz/k_rho direction
            (kz_cross, fz_cross) tuple(float): kz and fz where crossing found
            (k_rho_cross, f_rho_cross) tuple(float) same for k_rho and f_rho
        """

        # first look along kz:
        ve = beta*const.c  # speed of light
        m_z = (fz2-f)/(kz2-kz)  # gradient in kz direction
        m_rho = (f_rho2-f)/(k_rho2-k_rho)  # gradient in k_rho direction
        # electron speed components
        v_rho, v_z = ve*direction[0], ve*direction[1]  
        v_abs = (v_rho**2 + v_z**2)**0.5

        cz = f - m_z*kz  # f intercept constant in f=m*k+c
        c_rho = f - m_rho*k_rho
        # first look at kz direction
        if abs(m_z - v_z) < 1e-15*abs(v_rho*k_rho - cz):
            z_found = False  # m -> +-infinity
            kz_cross = fz_cross = None
        else:
            kz_cross = (v_rho*k_rho - cz)/(m_z - v_z)
            fz_cross = kz_cross*m_z + cz
            z_found = True
        z = (kz_cross, fz_cross)

        if abs(m_rho - v_rho) < 1e-20*abs(v_z*kz - c_rho):
            rho_found = False  # m -> +-infinity
            k_rho_cross, f_rho_cross = None, None
        else:
            k_rho_cross = (v_z*kz - c_rho)/(m_rho - v_rho)
            f_rho_cross = k_rho_cross*m_rho + c_rho
            rho_found = True
        rho = (k_rho_cross, f_rho_cross)

        if rho_found:  # check if in range that interpolation is valid
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

    def analyze_error(self, band, wl_range=[250.e-9, 500.e-9], 
                     theory_compare=True, wavelength_range=True):
        """Calculate average cherenkov angle and error from angle against
        wavelength data. Then have option to compare to effective medium
        theory and calculate for different wavelength ranges."""
        # TODO: make it obvious how to calculate neff - 
        # sqrt(kx*kx+ky*ky+kz*kz)**2./k0 (do i use k_rho or kz in numerator)
        kz = self.data['default'][band]['cherenkov']['kz']
        k_rho = self.data['default'][band]['cherenkov']['k_rho']
        f = self.data['default'][band]['cherenkov']['frequency']
        d_rho, dz = self.data['default'][band]['cherenkov']['direction']
        adj_for_e_diretion = np.arctan(dz/(d_rho+1e-20))
        theta = np.arctan(kz/(k_rho+1e-20)) - adj_for_e_diretion
        # then compute outside angle
        np.tan(theta)
        # wl = 2*np.pi*3.e8/f
        # wl = 2.*np.pi/(kz**2.+k_rho**2.+1e-7)**0.5
        wl = const.c/f
        # print(print(wl)
        # print(f)
        pos_th, pos_wl, mean, err = \
            self._calc_err(theta, wl, wl_range)
        neg_th, neg_wl, neg_mean, neg_err = \
            self._calc_err(theta, wl, wl_range, sign=-1)
        self.data['default'][band]['cherenkov']['angle'] = theta
        self.data['default'][band]['cherenkov']['pos'] = {'angle': pos_th}
        self.data['default'][band]['cherenkov']['pos']['wavelength'] = pos_wl
        self.data['default'][band]['cherenkov']['pos']['error'] = err
        self.data['default'][band]['cherenkov']['pos']['average'] = mean
        self.data['default'][band]['cherenkov']['neg'] = {'angle': neg_th}
        self.data['default'][band]['cherenkov']['neg']['wavelength'] = neg_wl
        self.data['default'][band]['cherenkov']['neg']['error'] = neg_err
        self.data['default'][band]['cherenkov']['neg']['average'] = neg_mean
        self.data['default'][band]['cherenkov']['wavelength'] = wl
