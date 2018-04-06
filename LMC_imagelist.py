"""
To create a list of sky images and choose runs only with decent normalisation
"""

import matplotlib.pyplot as plt
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
from gammapy.image import SkyImage, IACTBasicImageEstimator, SkyImageList
from gammapy.background import FoVBackgroundEstimator
from gammapy.detect import TSImageEstimator
from astropy.convolution import Gaussian2DKernel
from photutils.detection import find_peaks

import iminuit


plt.ion()

name="LMC"
datastore = DataStore.from_dir("/Users/asinha/HESS_newbkg")
src=SkyCoord.from_name(name)
sep=SkyCoord.separation(src,datastore.obs_table.pointing_radec)
Radius=10
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
                           nxpix=400, nypix=400, binsz=0.05,
                           xref=src.galactic.l.value, yref=src.galactic.b.value,
                           coordsys='GAL', proj='TAN',
                           )

print(ref_image)

#list of detected sources

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


images=image_estimator.run_indiv(mylist)
backnorm=[]
for im in images:
    backnorm.append(im["background"].meta['NORM'])

#now, fit a gaussian and ignore 5% of the runs...


class loglikelihood:
    def __init__(self,data1D, func):
        self.data = data1D
        self.pdf = func

        # take function signature
        func_signature = iminuit.describe(func)

        # overwrite func_code to provide func signature
        self.func_code = iminuit.util.make_func_code(func_signature[1:])
 #       self.func_defaults = None # Not sure why we need this

    def __call__(self,*arg):
        res = np.log(self.pdf(self.data,*arg))
        tmp =  -res.sum()
        return tmp

def gaussian(x, mu = 0., sigma = 1.):
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#mle = loglikelihood(backnorm,gaussian)
b2=np.asarray(backnorm)
b1=b2[b2>0.4]
mle = loglikelihood(b1,gaussian)

m = iminuit.Minuit(mle,mu=1.0,sigma=0.2)
m.migrad()
m.values
plt.hist(backnorm,normed=True,alpha=0.5)
xplot = np.linspace(0.3,1.3,500)
plt.plot(xplot,gaussian(xplot,**m.values),color='r')
ranges=[m.values['mu']-2.0*m.values['sigma'],m.values['mu']+2.0*m.values['sigma']]
for r in ranges:
    plt.axvline(x=r,color='k', linestyle='--')
plt.xlabel("Background normalisation")

condlist=(b2>ranges[0]) & (b2<ranges[1])
acc_b=b2(condlist)
myid1=np.asarray(myid)
acc_runs=myid1[condlist]


