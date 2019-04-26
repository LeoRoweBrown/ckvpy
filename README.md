# ckvpy
Python analysis package for wave optics simulation data, speficially for periodic structures that exhibit Cerenkov-like behaviour.

# User guide
For 2D models:
```import ckvpy
a = ckvpy.Analyze2D(<csv path>, headers=<correct order of columns as list of strings> (optional))
```
To plot Cherenkov angle and chromatic error against unit cell size
```
a.data.find_angle(wl_range=[<wavelength range in m>] (optional))
a.a_plot(filename (optional), modelname (optional, for graph title), band (optional))
```
To plot wavelength against Cherenkov angle
```
a.full_plot(filename (optional), modelname (optional, for graph title), band='0' (optional))
```
To compare to Maxwell Garnett and volume-weighted average theory for refractive index inside crystal
```
a.compare_sio2(ratio_2d=0.1 (optional), modelname (optional), filename= (optional),
               index="sio2" (optional), bands=[None] (optional), a_s=[None] (optional)))
```

# Planned refactor:
Analysis2D and Analysis3D no longer inherit from CSVLoader, but CSVLoader instance is created and passes
data dictionary to create dataAnalysis instance where things like sorting and Cherenkov angle and Chromatic
error calculations take place.

# Another planned feature
inside analysis.py, since to access data you must get the instance data from the data object which is called
self.data in analyze2D and analyze3D, this means self.data.data is typed. May override __getitem__ and __setitem__
to allow direct indexing of the object such that self.data[key] does self.data.data[key], but this hides behaviour?