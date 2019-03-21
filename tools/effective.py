import numpy as np

def index(E_matrix, E_inclusion, ratio):
    """
    variables:
    E_matrix: (float) dielectric constant of matrix material
    E_inclusion: (float) dielectric constant of inclusion material
    ratio: (float) volume ratio between inclusion and overall volume (matrix)
    returns (float) effective dielectric constant
    """
    if type(E_matrix) is np.ndarray:
        E_inclusion = np.ones_like(E_matrix)*E_inclusion
    elif type(E_inclusion) is np.ndarray:
        E_matrix = np.ones_like(E_inclusion)*E_matrix
    elif type(E_inclusion) is float and type(E_matrix) is float:
        pass
    else:
        raise TypeError("Values can be floats and numpy arrays only")

    return E_matrix*(2*ratio*(E_inclusion-E_matrix) + E_inclusion +\
        2*E_matrix)/(2*E_matrix + E_inclusion - ratio*(E_inclusion-E_matrix))

def ratio(E_matrix, E_inclusion, E_effective):
    if type(E_inclusion) is not float or type(E_matrix) is not float \
        or type(E_effective) is not float:
        raise TypeError("Values can be floats only")

    return (E_effective*(2*E_matrix+E_inclusion) - \
        E_matrix*(E_inclusion+2*E_matrix))/(E_effective*(E_inclusion-E_matrix)\
        + 2*E_matrix*(E_inclusion-E_matrix))

