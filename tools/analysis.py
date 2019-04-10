import numpy as np 

class dataAnalysis(object):
    """Class to handle wavelength cuts, sorting, angle finding etc."""
    def __init__(self, data):
        self.data_dict = data

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