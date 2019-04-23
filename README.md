# ckvpy
Python analysis package for wave optics simulation data, speficially for periodic structures that exhibit Cerenkov-like behaviour.

# Planned refactor:
Analysis2D and Analysis3D no longer inherit from CSVLoader, but CSVLoader instance is created and passes
data dictionary to create dataAnalysis instance where things like sorting and Cherenkov angle and Chromatic
error calculations take place.

# Another planned feature
inside analysis.py, since to access data you must get the instance data from the data object which is called
self.data in analyze2D and analyze3D, this means self.data.data is typed. May override __getitem__ and __setitem__
to allow direct indexing of the object such that self.data[key] does self.data.data[key], but this hides behaviour?