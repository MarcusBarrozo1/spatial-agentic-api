import numpy as np

def calculate_ndvi(red_band, nir_band):
    red_band = red_band.astype(float)
    nir_band = nir_band.astype(float)
    
    denominator = (nir_band + red_band)
    ndvi = np.divide(
        (nir_band - red_band), 
        denominator, 
        out=np.zeros_like(nir_band), 
        where=denominator != 0
    )
    return ndvi