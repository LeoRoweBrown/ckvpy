import csv
import math
import numpy as np
import json

class CSVLoader(object):
    def __init__(
                self, datafile, file_suffix='undefined', headers=[],
                sort_by='undefined'):
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
        self.path = None
        self.sort_by = sort_by
        self.col = {}
        self.num_bands = {}

        print("Using CSVLoader")

        for n, header in enumerate(headers):
            if header is 'skip':
                continue
            self.col[header] = n
        print(self.col)
        print("sorting by", self.sort_by)

        if type(file_suffix) is list and file_suffix[0] is not 'undefined':
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
        elif type(file_suffix) is str or file_suffix[0] is 'undefined':
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
                    # root key i.e. self.data[root]
                    root = '%.3g' % float(row[self.col[self.sort_by]])
                except KeyError:
                    raise KeyError("Chosen sort_by key does not exist in list"
                                   "of headers for this data set")
                except ValueError:
                    print("Unable to convert string to float ignoring line: ")
                    print(row)
                    continue
                except IndexError:
                    print("Using supplied value for root key from"
                          "file_suffix' since it is not found in file, or"
                          "table headings are formatted incorrectly")
                    root = self.file_suffix[0]

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
            for key in self.col:
                if key is self.sort_by:
                   continue
                self.data[root][key] = []
                b = 0
                reshape = False
                while b < num_bands:
                    if key is self.sort_by:
                        continue
                    data_dump = [line[self.col[key]]
                            for line in self.data[root]['raw']
                            if int(line[self.col['band']]) == b+1]
                    
                    if self.sort_by is 'band': # for bz2d.py
                        b = int(root) - 1 # dont keep appending data
                        self.data[root][key] = data_dump
                    else:
                        self.data[root][key].append(data_dump)
                    b += 1
                # avoid storing param key twice
                # e.g. avoid data['1.00E-07']['1.00E-07']

                # split data into bands (sub lists)
                # print("num", num_bands)
                # self.data[root][key] = [None]*num_bands
                
            self.data[root].pop('raw')

    def sort_data(self, key, direction=1):
        """sort ascending according to key, e.g. key = 'wavelength'"""
        for root in self.data:
            for b in range(self.num_bands[root]):
                data = self.data[root]
                # print("Sorting", key)
                sort_index = np.argsort(data[key][b])
                if direction == -1:
                    sort_index = sort_index[:-1]
                # first check if same number of sublists
                for param in self.data[root]:
                    if len(self.data[root][param]) != \
                        len(self.data[root][key]):
                            continue
                    # check same data length
                    if len(self.data[root][param][b]) == \
                        len(self.data[root][key][b]):
                            # print("sorting", param)
                            self.data[root][param][b] = \
                            [data[param][b][ind] for ind in sort_index]

    def save_data(self, name):
        with open(name, 'w') as f:
            json.dump(self.data, f)