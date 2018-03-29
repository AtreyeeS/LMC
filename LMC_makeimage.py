import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gammapy
import numpy as np
import astropy
import regions
import sherpa
import uncertainties
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.image import SkyImage, IACTBasicImageEstimator
from gammapy.background import RingBackgroundEstimator, ReflectedRegionsBackgroundEstimator, FoVBackgroundEstimator
from gammapy.utils.energy import EnergyBounds
from gammapy.detect import TSImageEstimator

plt.ion()

name="LMC"
datastore = DataStore.from_dir("/Users/asinha/HESS_newbkg") 
src=SkyCoord.from_name(name)
sep=SkyCoord.separation(src,datastore.obs_table.pointing_radec)
Radius=8
srcruns=(datastore.obs_table[sep<Radius*u.deg]) 
obsid=srcruns['OBS_ID'].data
srclist=datastore.obs_list(obsid)

low_exp=[]
for o1 in srclist:
    if o1.observation_live_time_duration.value<900.0:
        low_exp.append(o1.obs_id)

myid=list(set(obsid)-set(low_exp))
mylist=datastore.obs_list(myid)

total_time=0
for o1 in mylist:
    total_time=total_time+o1.observation_live_time_duration

print("%s observations within %.1f deg from target (%.2f,%.2f)"%(len(mylist),Radius,src.galactic.l.value,src.galactic.b.value))
print("Total observation duration is",total_time.to(u.hour))

ref_image = SkyImage.empty(
    nxpix=800, nypix=800, binsz=0.01,
    xref=src.galactic.l.value, yref=src.galactic.b.value,
    coordsys='GAL', proj='TAN',
)

print(ref_image)

#list of detected sources

src_names=["LMC N132D","30 Dor C","LHA 120-N 157B","LMC P3"]
l=[280.31,279.60, 279.55, 277.73]*u.deg
b=[-32.78,-31.91, -31.75,-32.09 ]*u.deg

srctab=Table([src_names,l,b],names=("src_names","longitude","latitude"),meta={'name': 'known source'})

off_regions=[]

for s in srctab:
    circle=CircleSkyRegion(center=SkyCoord(s['longitude'], s['latitude'], unit='deg', frame='galactic'), radius=0.2 *u.deg)
    off_regions.append(circle)

exclusion_mask = ref_image.region_mask(off_regions[0])
exclusion_mask.data = 1 - exclusion_mask.data
for off in off_regions[1:]:
    exclusion_mask.data -= ref_image.region_mask(off)

exclusion_mask.data[exclusion_mask.data<0]=0    #overlapping pixels go to negative values. Fixing them to zero
exclusion_mask.plot(add_cbar=True)

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

images = image_estimator.run(mylist)
images.names

