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

comp_image = image_estimator.run(mylist)
comp_image.names

images=comp_image

print("Total number of counts",images[0].data.sum())
print("Total number of excess events",images[3].data.sum())
    
fermi=SkyImage.read("FDA16.fits")
fermi_rep=fermi.reproject(ref_image)
    
#Make sure that the exclusion mask properly excludes all sources
    
masked_excess = SkyImage.empty_like(images['excess'])
masked_excess.data = comp_image['excess'].data * exclusion_mask.data
masked_excess_cutout=masked_excess.cutout(position=SkyCoord(src.galactic.l.value, src.galactic.b.value, unit='deg', frame='galactic'),size=(9*u.deg, 9*u.deg))
masked_excess_cutout.show(add_cbar=True)


#masked_excess.plot(cmap='viridis',stretch='sqrt',add_cbar=True)
#plt.hist(masked_excess.data.flatten(),bins=100,log=True)
    
#smooth
    
excess_smooth=comp_image[3].smooth(radius=0.5*u.deg)
excess_smooth.show(add_cbar=True)
counts_smooth=comp_image[3].smooth(radius=0.5*u.deg)
#counts_smooth.show()
    
#cutout
    
excess_cut=excess_smooth.cutout(position=SkyCoord(src.galactic.l.value, src.galactic.b.value, unit='deg', frame='galactic'),size=(9*u.deg, 9*u.deg))
#excess_cut.plot(cmap='viridis',add_cbar=True,stretch='sqrt')
                                    
counts_cut=counts_smooth.cutout(position=SkyCoord(src.galactic.l.value, src.galactic.b.value, unit='deg', frame='galactic'),size=(9*u.deg, 9*u.deg))

#counts_cut.plot(cmap='viridis',add_cbar=True,stretch='sqrt')

fermi_cutout=fermi_rep.cutout(position=SkyCoord(src.galactic.l.value, src.galactic.b.value, unit='deg', frame='galactic'),size=(9*u.deg, 9*u.deg))

exp_cut=comp_image[1].cutout(position=SkyCoord(src.galactic.l.value, src.galactic.b.value, unit='deg', frame='galactic'),size=(9*u.deg, 9*u.deg))

#fig, ax, _ =excess_cut.plot(cmap='viridis',add_cbar=True,stretch='sqrt')
#ax.contour(fermi_cutout.data, cmap='Blues')

#now, remove runs with poor normalisation.

kernel = Gaussian2DKernel(2.5, mode='oversample')
estimator = TSImageEstimator()
result = estimator.run(images, kernel)
print('TS map computation took {0:.2f} s'.format(result.meta['runtime']))

fig_ts = plt.figure(figsize=(18, 4))
ax_ts = fig_ts.add_axes([0.1, 0.1, 0.9, 0.9], projection=images['counts'].wcs)
ax_ts.imshow(result['sqrt_ts'], cmap='afmhot', origin='lower', vmin=0, vmax=10)
ax_ts.coords['glon'].set_axislabel('Galactic Longitude')
ax_ts.coords['glat'].set_axislabel('Galactic Latitude')

# Plot flux map (in units of m^-2 s^-1 TeV^-1)
fig_flux = plt.figure(figsize=(18, 4))
ax_flux = fig_flux.add_axes([0.1, 0.1, 0.9, 0.9], projection=images['counts'].wcs)
ax_flux.imshow(result['flux'], cmap='afmhot', origin='lower', vmin=0, vmax=1E-9)
ax_flux.coords['glon'].set_axislabel('Galactic Longitude')
ax_flux.coords['glat'].set_axislabel('Galactic Latitude')


# Plot number of iterations of the fit per pixel
fig_iter = plt.figure(figsize=(18, 4))
ax_iter = fig_iter.add_axes([0.1, 0.1, 0.9, 0.9], projection=images['counts'].wcs)
ax_iter.imshow(result['niter'], cmap='afmhot', origin='lower', vmin=0, vmax=20)
ax_iter.coords['glon'].set_axislabel('Galactic Longitude')
ax_iter.coords['glat'].set_axislabel('Galactic Latitude')

sources = find_peaks(
    data=result['sqrt_ts'].data,
    threshold=10,
    wcs=result['sqrt_ts'].wcs,
)
sources

result_cutout=result['sqrt_ts'].cutout(position=SkyCoord(src.galactic.l.value, src.galactic.b.value, unit='deg', frame='galactic'),size=(9*u.deg, 9*u.deg))
result_cutout.show(add_cbar=True)

plt.gca().scatter(
    sources['icrs_ra_peak'], sources['icrs_dec_peak'],
    transform=plt.gca().get_transform('icrs'),
    color='none', edgecolor='white', marker='o', s=600, lw=1.5,
)

exclusion_cutout=exclusion_mask.cutout(position=SkyCoord(src.galactic.l.value, src.galactic.b.value, unit='deg', frame='galactic'),size=(9*u.deg, 9*u.deg))
exclusion_cutout.show()

fig, ax, _ = result_cutout.plot(add_cbar=True,stretch='sqrt')
plt.gca().scatter(
     sources['icrs_ra_peak'], sources['icrs_dec_peak'],
     transform=plt.gca().get_transform('icrs'),
     color='none', edgecolor='red', marker='o', s=600, lw=1.5,
 )

fermi_exp=fermi_cutout.data*exp_cut.data
ax.contour(fermi_exp)
ax.contour(fermi_cutout.data, cmap='Blues')



#now, add the source and re-do


for s in sources:
    obj=SkyCoord(s["icrs_ra_peak"],s["icrs_dec_peak"],unit="deg")

    circle=CircleSkyRegion(center=SkyCoord(obj.galactic.l.value,obj.galactic.b.value, unit='deg', frame='galactic'), radius=1.0 *u.deg)
    off_regions.append(circle)


exclusion_mask = ref_image.region_mask(off_regions[0])
exclusion_mask.data = 1 - exclusion_mask.data
for off in off_regions[1:]:
    exclusion_mask.data -= ref_image.region_mask(off)

exclusion_mask.data[exclusion_mask.data<0]=0    #overlapping pixels go to negative values. Fixing them to zero
exclusion_mask.plot(add_cbar=True)








