import csv
import math
import numpy as np
import json
import scipy.constants as const

# TODO: Remove analysis methods like calc_err from this loader class
class CSVLoader(object):
    def __init__(
                self, datafile, file_suffix='default', headers=[],
                sort_by='default'):
        """
        Base class to dump and analyse Cerenkov data tables from COMSOL.
        Aim is to be adaptable to different datasets.

        Params:
        datafile (str): path of raw csv data
        file_suffix (list<str/int>): Split mode: list of data file suffixes
            for when there is data separated into different files for
            different parameter values. The datafile supplied is the prefix
            i.e. foo_ for foo_100 *without* the .csv extension.
            Will be automatically read and sorted in the data dictionary.
            Single mode: when file_suffix is not supplied or paramter is a
            single string. This only processes data for one param value.
        headers (list<str>): list of paramters featured in data table in order
            of appearance (left to right). Use 'skip' to ignore a column.
            Example: ['band', 'k', 'wavelength', 'angle', 'a'].
        sort_by (str): paramter in headers by which the data dictionary is
            sorted by - sort_by determines the root key. That is the
            outermost index e.g. if sort_by = 'a',
            data = { '100nm' : { <data> } '200nm' : { <data> } ... }.

        Attributes:
        format (str): format of the data: 'split' means separate files for
            each file_suffix paramter and is for automatically looping
            over a set of params, 'single' means only one parameter is being
            considered and 'merged' means data for different paramters is in
            one file. This is automatically determined.
        data (dict[key = paramter: value = list[float] ]): Data dictionary
            sorted by sort_by paramter as the 'root' key.
        path (str): path of datafile, set in read(), used for automatic graph
            titles if no model name supplied.
        col (dict[key = variable name : column index]): dictionary of column
            numbers for each variable in csv file - for dealing with
            differently formatted data tables.
        prefix (str): Used in split mode, e.g. foo_ in foo_100.csv
        num_bands (dict[key = root]: value = number of bands (int))

        """

        self.prefix = None
        self.data = {}  # dictionary of data, keys are unit cell size
        self.path = datafile
        self.sort_by = sort_by
        self.col = {}
        self.num_bands = {}
        # headers += 'default' # to handle case when root key is 'default'

        if datafile[-5:] == '.json':
            print("Loading JSON file")
            with open(datafile, 'r') as j:
                self.data = json.load(j)
            return

        print("Using CSVLoader")

        for n, header in enumerate(headers):
            if header is 'skip':
                continue
            self.col[header] = n
        print(self.col)
        if self.sort_by is 'band':
            raise KeyError("Sorting data by band as root key not supported")
        print("sorting by", self.sort_by)

        if type(file_suffix) is list and file_suffix[0] is not 'default':
            self.format = 'split'
            self.prefix = datafile
            self.file_suffix = file_suffix
            print("List of unit cell sizes has been supplied. These will used"
                  "in find_angle automatically unless loop=False.")
            print("NOTE: for the entire dataset (wavelength vs angle), you "
                  "must choose desired value of 'file_suffix' to read by the "
                  "method read(<path>). Otherwise, the first value in the "
                  "list will be used.")
            datafile += str(file_suffix[0]) + ".csv"
        elif type(file_suffix) is str or file_suffix[0] is 'default':
            self.file_suffix = [file_suffix]
            self.format = 'single'
        try:
            self.read(datafile)  # use first value in list by default
            if len(self.file_suffix) > 0:
                self.format = 'merged'
        except FileNotFoundError as f:
            raise f
        except OSError:
            raise OSError(
                "Failed to parse filename, ensure format: "
                "'..\\dir\\subdir\\file[.csv]' or '../dir/subdir/file[.csv]'")
        except:
            raise IOError(
                    "Failed to load raw data into dictionary, check the "
                    "format. The variables stored in the dictionary can be "
                    "changed using headers = ['col1', 'col2' ...] where the "
                    "elements are table headings.")

    def _get_raw(self, path):
        with open(path) as file:
            print("Reading file", path)
            reader = csv.reader(file)
            for n_rows, row in enumerate(reader):
                if '%' in ''.join(row):  # skip headings etc.
                    print("ommitting", row)
                    continue
                # if self.format is 'merged' or self.format is 'single':
                try:
                    # root key i.e. self.data[root][band]
                    root = '%.3g' % float(row[self.col[self.sort_by]])
                except KeyError:
                    if self.sort_by == 'default':
                        # print("Using default as root key")
                        root = 'default'
                    else:
                        raise KeyError("Chosen sort_by key does not exist in "
                                       "list of headers for this data set")
                except ValueError:
                    print("Unable to convert string to float ignoring line: ")
                    print(row)
                    continue
                except IndexError:
                    print("column does not exist in data "
                          "and the file_suffix is being used as the key.")
                    root = self.file_suffix[0] # is 'default' by default 
                    self.sort_by = root

                # for repeated calls of _get_raw with different root keys
                # e.g. for a, '1.00E-07', '1.50E-07'
                if root not in self.data:
                    self.data[root] = {}
                    self.data[root]['raw'] = []
                self.data[root]['raw'].append(
                    [float(complex(col.replace('i', 'j')).real)
                    for col in row]
                    )  # short CSV, just append
        return n_rows

    def read(self, path):
        """Open CSV and dump into list of lists, 
        [band number, k, wavelength, Cherenkov angle]
        
        Params:
        Path (str): file path of CSV file
        """
        self.path = path
        # split into bands, etc.
        #print("preloop, param =", self.file_suffix)
        for suffix in self.file_suffix: # for split mode
            if self.format is 'split':
                path = self.prefix + str(suffix) + ".csv"

            n_rows = self._get_raw(path)
            if self.format is 'single':
                break # not in split mode so only need one iteration
        # b is index for band, sublists that represnt data for each band
        # unless b is the root key (is self.sort_by)
        for root in self.data.keys(): # fill each entry according to root key
            num_bands = int(max([line[self.col['band']] 
                    for line in self.data[root]['raw']]))
            self.num_bands[root] = num_bands
            b = 0
            while b < num_bands:
                band = str(b)
                self.data[root][band] = {}
                for key in self.col:
                    if key is self.sort_by:
                        continue
                    data_dump = [line[self.col[key]]
                        for line in self.data[root]['raw']
                        if int(line[self.col['band']]) == b+1]
                    self.data[root][band][key] = data_dump
                # if 'n' not in self.col:
                #     if 'kx' in self.col and 'ky' in self.col and \
                #         'kz' in self.col and 'k_abs' not in self.col:
                #         self.data[root][band]['k_abs'] = \
                #             (kx**2. + ky**2. + kz**2.)**0.5
                #         n = self._compute_n()
                #         self.data[root][band]['n'] = n
                b += 1
                print(self.data[root][band].keys())
            self.data[root].pop('raw')

    def _compute_n(self):  # unecessary? 
        w1 = 2*np.pi*np.array(self.data[root][band][freq])[:-1]
        w2 = 2*np.pi*np.array(self.data[root][band][freq])[1:]
        dw = w2-w1
        k1 = 2*np.pi*np.array(self.data[root][band]['k_abs'])[:-1]
        k2 = 2*np.pi*np.array(self.data[root][band]['k_abs'])[1:]
        dk = k2-k1
        n = const.c/(dw/dk)
        return n

    def sort_data(self, key, subkeys=None, direction=1):
        """sort ascending according to key, e.g. key = 'wavelength'
        Subkeys allow for sorting nested data"""
        for root in self.data:
            num_bands = self.num_bands[root]
            for band in self.data[root]:
                data = self.data  # reset data to entire dict
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
                    # print(sorted(self.data[root][band][key])==self.data[root][band][key])

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
        th_pos, wl_pos = self.wl_cut(a, band)  # take positive angles
        self.data[a][band]['angle'] = th_pos
        self.data[a][band]['wavelength'] = wl_pos

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

    def wl_cut(self, a='default', band='0', param_key=None, 
              wl_range=[0.,1e10], sign=1):
        """Take cut of data based on wavelength range. Default behaviour
        removes negative angles"""
        wl = self.data[a][band]['wavelength']
        theta = self.data[a][band]['angle']
        param = self.data[a][band][param_key]
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

    def save_data(self, name):
        with open(name, 'w') as f:
            json.dump(self.data, f)

    def jsonify(self):
        raise NotImplementedError
        self._search(self.data)

    def _search(data):
        raise NotImplementedError
        for key in data:
            if type(data[key]) is dict:
                _search(data[key])
            elif type(data['key']) is np.ndarray:
                data['key'] = data.tolist()
