# ckvpy
Python analysis package for wave optics simulation data, speficially for periodic structures that exhibit Cerenkov-like behaviour.

# Planned refactor:
Analysis2D and Analysis3D no longer inherit from CSVLoader, but CSVLoader instance is created and passes
data dictionary to create dataAnalysis instance where things like sorting and Cherenkov angle and Chromatic
error calculations take place.