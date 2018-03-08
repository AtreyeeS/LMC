
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Ring2DKernel, Tophat2DKernel
from astropy.visualization import simple_norm

from gammapy.data import DataStore
from gammapy.image import SkyImage, SkyImageList
from gammapy.detect import KernelBackgroundEstimator as KBE

#from gammapy.irf import EnergyDispersion, EnergyDispersion2D, EffectiveAreaTable, EffectiveAreaTable2D

# Setup the logger
import logging
logging.basicConfig()
log = logging.getLogger('gammapy.spectrum')
log.setLevel(logging.ERROR)


#choose the obsevartion
name="LMC"
datastore = DataStore.from_dir("$HESS_DATA") 
src=SkyCoord.from_name(name)
sep=SkyCoord.separation(src,datastore.obs_table.pointing_radec)
srcruns=(datastore.obs_table[sep<5.0*u.deg]) #2 deg because of HESS field of view.. choose accordingly 
obsid=srcruns['OBS_ID'].data

# Define obs parameters
#lo_threshold = 0.1 * u.TeV
#hi_threshold = 60 * u.TeV


