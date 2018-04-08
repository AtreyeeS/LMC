"""
to create the list of SkyImageLists
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy
import regions
import sherpa
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.image import SkyImage, IACTBasicImageEstimator, SkyImageList
from gammapy.background import FoVBackgroundEstimator

from importlib import reload

datastore = DataStore.from_dir("/Users/asinha/HESS_newbkg")
name="LMC"
src=SkyCoord.from_name(name)
myid=np.loadtxt("LMC_id.txt")
mylist=datastore.obs_list(myid)

ref_image = SkyImage.empty(
                           nxpix=400, nypix=400, binsz=0.05,
                           xref=src.galactic.l.value, yref=src.galactic.b.value,
                           coordsys='GAL', proj='TAN',
                           )
src_names=["LMC N132D","30 Dor C","LHA 120-N 157B","LMC P3"]
l=[280.31,279.60, 279.55, 277.73]*u.deg
b=[-32.78,-31.91, -31.75,-32.09 ]*u.deg
radius=[0.2,0.2,0.5,0.2]*u.deg

srctab=Table([src_names,l,b,radius],names=("src_names","longitude","latitude","radius"),meta={'name': 'known source'})

off_regions=[]

for s in srctab:
    #circle=CircleSkyRegion(center=SkyCoord(s['longitude'], s['latitude'], unit='deg', frame='galactic'), radius=0.2*u.deg)
    circle=CircleSkyRegion(center=SkyCoord(s['longitude'], s['latitude'], unit='deg', frame='galactic'), radius=s['radius']*u.deg)
    off_regions.append(circle)

exclusion_mask = ref_image.region_mask(off_regions[0])
exclusion_mask.data = 1 - exclusion_mask.data
for off in off_regions[1:]:
    exclusion_mask.data -= ref_image.region_mask(off)


exclusion_mask.data[exclusion_mask.data<0]=0   
bkg_estimator = FoVBackgroundEstimator()


emin=0.5 * u.TeV
emax=20 * u.TeV
offset_max=2.3 * u.deg

image_estimator = IACTBasicImageEstimator(
                                          reference=ref_image,
                                          emin=emin,
                                          emax=emax,
                                          offset_max=offset_max,
                                          background_estimator=bkg_estimator,
                                          exclusion_mask=exclusion_mask)


images=image_estimator.run_indiv(mylist)

fermi="FDA16.fits"
diffuse=SkyImage.read(fermi)
diffuse_rep=diffuse.reproject(ref_image)

#make exclusion and cutouts

exc_list=[]
diffuse_list=[]

for i in range(len(mylist)):
    exc_list.append(image_estimator._cutout_observation(exclusion_mask,mylist[i]))
    diffuse_list.append(image_estimator._cutout_observation(diffuse_rep,mylist[i]))

backnorm=[]
for im in images:
    backnorm.append(im["background"].meta['NORM'])

