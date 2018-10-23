import numpy as np
import astropy.units as u
from gammapy.spectrum.models import PowerLaw
from gammapy.image.models import SkyGaussian
from gammapy.cube.models import SkyModel, SkyModels
from gammapy.cube import MapEvaluator
from gammapy.maps import WcsGeom, WcsNDMap, Map

class MyLikelihood(object):
    """
        Class to compute the b_k and the likelihood function
    """
    
    def __init__(self, count_list, exposure_list, background_list, diffuse_list=None, exclusion_list=None):
        self.count_list = count_list
        self.exposure_list = exposure_list
        self.background_list = background_list
        self.exclusion_list = exclusion_list
        self.diffuse_list = diffuse_list #Diffuse emission template map... gaussian in this case
    
    def __call__(self, lon0, lat0, sig, amp):
        lon0 = lon0 * u.deg
        lat0 = lat0 *u.deg
        sig = sig * u.deg
        amp = amp/(u.cm *u.cm * u.s *u.TeV)
        return self.calc_res(lon0, lat0, sig, amp)
    
    
    def calc_res(self, lon0, lat0, sig, amp):
        """
            calculates the log likelihood function to minimise
            
            mu, sig = the mean and sigma of the diffuse emission
            
            Returns
            
            res: The log likelihood function
            
        """
        
        Sk_map, b = self.calc_bk(lon0, lat0, sig, amp)
        #print(b)
        res=0.0
        t1=0.0
        t2=0.0
        for k in range(len(b)):
            Nk = self.count_list[k].data
            #exp_map = self.exposure_list[k]
            Bk = self.background_list[k].data
            Sk = Sk_map[k].data
            
            v1 = (Sk + b[k]*Bk)
            v1_0 = (v1==0)
            ln_v1 = np.log(v1)
            ln_v1[v1_0] = 0.0 #To handle pixels values with 0 counts
            
            t1 = t1+np.sum(Nk * ln_v1)
            t2 = t2+np.sum(Sk + b[k]*Bk)
        #print(t1,t2, (alpha*Sk + b[k]*Bk))
        res = t1-t2
        return -res

    
    def calc_bk(self, lon0, lat0, sig, amp):
        """
            returns the computed b_k and the diffuse model template.
        """
        # Define sky model to fit the data
        ind = 2.0
        
        spatial_model = SkyGaussian(lon_0=lon0, lat_0=lat0, sigma=sig)
        spectral_model = PowerLaw(
            index=ind, amplitude=amp, reference="1 TeV")
        model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
        
        # For simulations, we can have the same npred map
        exp = self.exposure_list[0]
        evaluator = MapEvaluator(model=model, exposure=exp)
        npred = evaluator.compute_npred()
        geom = exp.geom
        diffuse_map = WcsNDMap(geom, npred) #This is Sk

        b_k = []
        Sk_list = []
        Bk = self.background_list[0].data
        
        for count in self.count_list:
           
            Sk = diffuse_map.data
            Nk = count.data
            
            not_has_exposure = ~(exp.data > 0)
            not_has_bkg = ~(Bk > 0)
            
            S_B = np.divide(Sk,Bk)
            S_B[not_has_exposure] = 0.0
            S_B[not_has_bkg] = 0.0
            
            #Sk is nan for large sep.. to be investigated. temp soln
            #if np.isnan(np.sum(S_B)):
            #    S_B=np.zeros(S_B.shape)
            
            
            delta=np.power(np.sum(Nk)/np.sum(Bk),2.0) - 4.0 * np.sum(S_B)/np.sum(Bk)
            #print(np.sum(Nk),np.sum(Bk),np.sum(Sk),np.sum(S_B), delta)
            #print("delta is %f for obs no %s",delta,k)
            #bk1=(np.sum(Nk)/np.sum(Bk) - np.sqrt(delta))/2.0
            bk2 = (np.sum(Nk)/np.sum(Bk) + np.sqrt(delta))/2.0
            b_k.append(bk2)
            Sk_list.append(diffuse_map)
        
        return Sk_list, b_k
