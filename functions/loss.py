import numpy as np

def phi(z: np.ndarray):
    """
    Smoothed hinge loss (phi) et sa dérivée (phi').
    Entrée:
      z: array numpy (1D/2D ok)
    Sorties:
      u = phi(z), v = phi'(z)  (même shape que z)
    """

    # S'assurer que z est un tableau numpy
    z = np.asarray(z, dtype=np.float64)
    u = np.zeros_like(z)  
    v = np.zeros_like(z)   


    # Définir les masques pour chaque région
    M1 = z <= -1                # Région 1 : z <= -1 (perte nulle)
    M2 = (z > -1) & (z < 0)     # Région 2 : -1 < z < 0 (zone de transition lissée)
    M3 = z >= 0                 # Région 3 : z >= 0 (perte linéaire)

    # Region 1 : z <= -1 already initialized to zero

    # Region 2 : -1 < z < 0
    u[M2] = 0.5 * (1 + z[M2])**2   
    v[M2] = 1 + z[M2]              

    # Region 3 : z >= 0
    u[M3] = 0.5 + z[M3]           
    v[M3] = 1.0                    

    # Return u = phi(z) and v = phi'(z)
    return u, v 

