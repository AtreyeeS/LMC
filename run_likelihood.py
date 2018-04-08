import iminuit
import astropy
import regions
import sherpa
import uncertainties
import astropy.units as u
from importlib import reload


import MyLikelihood
reload (MyLikelihood)
from MyLikelihood import *
mle=MyLikelihood(images, diffuse_list=diffuse_list, exclusion_list=exc_list)
mle.calc_bk(0.1)

image_list=images
count_list=[row[0] for row in image_list]
exposure_list=[row[1] for row in image_list]
background_list=[row[2] for row in image_list]
exclusion_list=exc_list
diffuse_list=diffuse_list

"""
k=11
count_map=count_list[k]
exp_map=exposure_list[k]
bkg_map=background_list[k]
Nk=count_map.data *  exclusion_list[k].data
Sk=diffuse_list[k].data * exp_map.data * exclusion_list[k].data
Bk=bkg_map.data * exclusion_list[k].data  #multiply with exposure?

not_has_exposure = ~(exp_map.data > 0)
not_has_bkg = ~(Bk > 0)
has_bkg=~not_has_bkg

S_B=np.divide(Sk,Bk)
S_B[not_has_exposure]=0.0
S_B[not_has_bkg]=0.0
"""
    


m=iminuit.Minuit(mle,alpha=0.01)
m.migrad()
m.values
m.errors


