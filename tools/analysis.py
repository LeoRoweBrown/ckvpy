import numpy as np
import scipy.constants as const
import json
import os
from matplotlib import pyplot as plt
import ckvpy.tools.photon_yield as photon_yield
import ckvpy.tools.effective as effective

class dataAnalysis(object):
    """Class to handle wavelength cuts, sorting, angle finding etc."""
    def __init__(self, data, path):
        self.data_dict = data
        self.path = path
        self._get_num_bands()
        self._rm_nan()
    
    def _get_num_bands(self):
        self.num_bands = {}
        for root in self.data_dict:
            i = 0
            for bands in self.data_dict[root]:
                i += 1
            self.num_bands[root] = i

    def _rm_nan(self):
        """Remove NaNs in data using a mask"""
        for root in self.data_dict:
            for band in self.data_dict[root]:
                final_mask = None
                for param in self.data_dict[root][band]:
                    data = self.data_dict[root][band][param]
                    band_len = len(self.data_dict[root][band]['band'])
                    if type(data) is list and len(data) == band_len:
                        nan_array = np.isnan(data)
                        # print(nan_array[-1])
                        # nan_list = [val is not 'nan' for val in data]
                        if final_mask is None:
                            final_mask = nan_array
                        final_mask = np.logical_or(final_mask, nan_array)
                final_mask = np.logical_not(final_mask)
                for param in self.data_dict[root][band]:
                    # do elementwise pop() instead of this strange conversion?
                    band_len = len(self.data_dict[root][band]['band'])
                    data = self.data_dict[root][band][param]
                    if type(data) is list and len(data) == band_len:
                        data = np.array(data)[final_mask].tolist()

    def save(self, name):
        with open(name, 'w') as f:
            json.dump(self.data_dict, f)

    def find_angle(self, wl_range=[250.0e-9, 500.0e-9], filename=None):
        """Get Cherenkov angles/chromatic error for wavelength range wl_range
        
        Params:
            wl_range list[float]: wavelength range that Cherenkov angle and
                chromatic error is calculated over
        Returns:
            tuple(float): Cherenkov angle average and range
        """
        for a in self.data_dict:
            for band in self.data_dict[a]:
                print("Finding angle for a =", a, "band", band)
                wl1, wl2, average, rnge = self.calc_err(wl_range, a=a, band=band)
                try:
                    a_ = float(a)
                except ValueError:
                    a_ = 0.
                array = np.array([wl1, wl2, average, rnge, float(a_)])
                self.data_dict[a][band]['cherenkov'] = array.tolist()  # json friendly
                # self.data_dict[a][band][str(wl1)+'-']
        # print(average, rnge)
        return average, rnge
        # dont return average and range if computed 
        # for multiple values of 'a', these are stored in file.
    
    def calculate_n_eff(self, method='gradient'):
        """method is 'gradient' or 'angle', TODO: may remove, redundant"""
        for root in self.data_dict:
            for band in self.data_dict[root]:
                data = self.data_dict[root][band]
                if 'n' in data:
                    print('refractive index already in data')
                    continue
                if 'kx' in data and 'ky' in data and 'ky' in data:
                    kx = np.array(data['kx'])
                    ky = np.array(data['ky'])
                    kz = np.array(data['kz'])
                    kabs = np.sqrt(kx*kx+ky*ky+kz*kz)
                    th_in = np.arctan(kz/np.sqrt(kx*kx+ky*ky))
                elif 'kz' in data and 'k_rho' in data:  # 3D
                    kz = np.array(data['kz'])
                    k_rho = np.array(data['k_rho'])
                    kabs = np.sqrt(k_rho*k_rho+kz*kz)
                    d_rho, dz = self.data_dict['default'][band]['direction']
                    if dz == 1: 
                        k_parallel = k_rho
                        k_perp = kz
                    elif d_rho == 1: 
                        k_parallel = kz
                        k_perp = k_rho
                    th_in = np.arctan(k_parallel/(k_perp+1e-20))
                else:
                    raise ValueError("No kx, ky and kz in dataset")
                f = np.array(data['frequency'])
                k0 = 2*np.pi*f/const.c # omega/c
                neff = kabs/k0
                wl_in = 2*np.pi/kabs

                wl_nan = np.isnan(wl_in)  # deal with NaNs
                th_nan = np.isnan(th_in)
                neff_nan = np.isnan(neff)
                nan_mask = np.logical_or(wl_nan, th_nan, neff_nan)
                nan_mask = np.logical_not(nan_mask)
                data['n_eff'] = neff[nan_mask].tolist()
                data['wl_in'] = wl_in[nan_mask].tolist()
                data['th_in'] = th_in[nan_mask].tolist()

    def calculate_n_mg(self, ratio, index="sio2"):
        for root in self.data_dict:
            for band in self.data_dict[root]:
                wl_in = self.data_dict[root][band]['wl_in']
                index_file = os.path.join(os.path.dirname(__file__),\
                    "..\\index\\"+ index + ".txt")
                wl_sio2, n_sio2 = np.loadtxt(index_file).T
                n_sio2_interp = np.interp(wl_in, wl_sio2, n_sio2)
                e_sio2 = n_sio2_interp*n_sio2_interp
                n_mg = np.sqrt(
                    effective.maxwell_garnett_index(1.0, e_sio2, ratio))
                self.data_dict[root][band]['n_mg'] = n_mg
    
    def average_n(self, wl_range):
        """finds average n_mg and n_eff (from data) in given wavelength range
        TODO: test"""
        for root in self.data_dict:
            for band in self.data_dict[root]:
                n_mg = self.data_dict[root][band]['n_mg']
                # find n where wl_range[0] < wavelength < wl_range[1]
                for r in range(len(wl_range)): # lower and upper range e.g.
                                               # wl_range = [250e-9, 500.e-9]
                    i = 0  # ith data value
                    w = 0. # wavelength
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
                            # final j decided on final iteration
                            if abs(wl1_diff) > abs(wl2_diff):
                                j = i
                            else:
                                j = i+1
                        i += 1  # next data point
                    if r == 0:  # for i1:i2 slice
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
        # print(wl_pos)

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
            print("Dataset doesn't seem to match desired wavelength "
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

    def wl_cut(self, root='default', band='0', wl_range=(0.,1e10),\
               sign=1, param_key=None, mutate=False):
        """Take cut of data based on wavelength range. Default behaviour
        removes negative angles if sign=1 or positive if sign=-1
        param_key: Keys other than wavelength to cut for:
            'all', None (angle), and any other paramter
        mutate: updates dicitonary with these values. Only valid when
            param_key is 'all'.
        """
        wl_nm_range = []
        param_nm_range = []
        list_param_nm_range = []
        print(wl_range)
        theta = self.data_dict[root][band]['angle'].copy()
        wl = self.data_dict[root][band]['wavelength'].copy()

        if param_key is 'all':
            len_wl = len(wl)
            param_key = [key for key in self.data_dict[root][band]\
                if len(self.data_dict[root][band][key]) == len_wl]
        if param_key is None:
            param_key = ['angle']
        if type(param_key) is str:
            param_key = [param_key]
        print('cutting for', param_key)
        for n, key in enumerate(param_key):
            param_nm_range = []
            for i, w in enumerate(wl):
                # print(theta[i], param[i])
                # print(len(wl), len(self.data_dict[root][band][key]))
                if w < wl_range[1] and w > wl_range[0] and sign*theta[i]>0:
                    # print("wavelength", w, w>wl_range[1])
                    print(w)
                    if n == 0: wl_nm_range.append(w)
                    param = self.data_dict[root][band][key][i]
                    param_nm_range.append(param)
            if mutate:
                print("Cutting dictionary values")
                self.data_dict[root][band][key] = param_nm_range
                print(key)
                print(np.mean(param_nm_range))
            list_param_nm_range.append(param_nm_range)

        if len(list_param_nm_range) == 1:
            return wl_nm_range, param_nm_range
        else:
            return wl_nm_range, list_param_nm_range

    def wl_rm(self, param_list=('all',), wl_range=(0.,1000.e-9)):
        """same as wl_cut, but mutates dictionary and takes multiple params
        TODO: This should be removed and combined with wl_cut and add 
        option to mutate dictionary? DEPRECATED"""
        for root in self.data_dict:
            for band in self.data_dict[root]:
                if param_list[0] is 'all':
                    len_wl = len(self.data_dict[root][band]['wavelength'])
                    param_list = [key for key in self.data_dict[root][band]\
                        if len(self.data_dict[root][band][key]) == len_wl]
                print(param_list)
                wl_data = None
                for param in param_list:
                    wl_data, param_data = \
                        self.wl_cut(root, band, wl_range, 1, param)
                    self.data_dict[root][band][param] = param_data
                    print(param, len(param_data))
                self.data_dict[root][band]['wavelength'] = wl_data

    def save_cherenkov(self, filename):
        """Save Cherenkov analysis data into a table"""
        matrix = np.empty((0,5))
        for a in self.data_dict:
            for b in self.data_dict[a]:
                line = np.array(self.data_dict[a][b]['cherenkov'])
                matrix = np.vstack((matrix, line))
        print(matrix)
        np.savetxt(filename, matrix)

    def calculate_yield(
        self, beta=0.999, L=100.e-6, wl_range=[250.e-9, 500.e-9]):
        """Compute photon yield using Frank-Tamm theory and effective 
        index. TODO: Currently uses outside angle to find n, rather
        than the correct n which is n_eff."""
        # raise NotImplementedError
        for root in self.data_dict:
            for band in self.data_dict[root]:
                theta = self.data_dict[root][band]['angle']
                f = self.data_dict[root][band]['frequency']
                wl, theta = self.wl_cut(root, band, wl_range)
                wl, f = self.wl_cut(root, band, wl_range, 'frequency')
                n = self.data_dict[root][band]['n_eff']
                n_p = photon_yield.compute(theta=theta, f=f, beta=0.999,
                                        L=L, n=n)
                if 'yield' not in list(self.data_dict[root][band]):
                    self.data_dict[root][band]['yield'] = {
                        'range': [],
                        'L': [],
                        'n_photons': []
                    }
                self.data_dict[root][band]['yield']['range'].append(wl_range)
                self.data_dict[root][band]['yield']['L'].append(L)
                self.data_dict[root][band]['yield']['n_photons'].append(n_p)
    
    def sort_data(self, key, subkeys=None, direction=1):
        """sort ascending according to key, e.g. key = 'wavelength'
        Subkeys allow for sorting nested data"""
        for root in self.data_dict:
            for band in self.data_dict[root]:
                data = self.data_dict  # reset data to entire dict
                if subkeys is None:
                    data = data[root][band]
                else:
                    data = data[root][band]
                    for sk in subkeys:
                        data = data[sk]
                # print("Sorting", key)
                sort_index = np.argsort(data[key])
                if direction == -1:
                    sort_index = sort_index[:-1]
                # first check if same number of sublists
                for param in data:
                    if len(data[param]) != \
                        len(data[key]):
                            continue
                    # check same data length
                    if len(data[param]) == \
                        len(data[key]):
                            # print("sorting", param)
                            data[param] = \
                            [data[param][ind] for ind in sort_index]
