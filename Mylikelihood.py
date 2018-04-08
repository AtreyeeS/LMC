import numpy as np
from gammapy.image import SkyImage, SkyImageList

class MyLikelihood(object):
    """
    Class to compute the b_k and do the compute the likelihood function
    """

    def __init__(self,image_list,diffuse_list=None,exclusion_list=None):
        self.count_list=[row[0] for row in image_list]
        self.exposure_list=[row[1] for row in image_list]
        self.background_list=[row[2] for row in image_list]
        self.exclusion_list=exclusion_list
        self.diffuse_list=diffuse_list



    def __call__(self,alpha):
        return self.calc_res(alpha)
        

    def calc_res(self,alpha):
        #calculates the log likelihood function to minimise 
        b = self.calc_bk(alpha)
        #print(b)
        res=0.0
        t1=0.0
        t2=0.0
        for k in range(len(b)):
            count_map=self.count_list[k]
            exp_map=self.exposure_list[k]
            bkg_map=self.background_list[k]
            Nk=count_map.data * self.exclusion_list[k].data
            Sk=self.diffuse_list[k].data * exp_map.data * self.exclusion_list[k].data
            
            #Sk is nan for large sep.. to be investigated. temp soln
            if np.isnan(np.sum(Sk)):
                Sk=np.zeros(Sk.shape)
                #continue

            Bk=bkg_map.data * self.exclusion_list[k].data
            
            v1=(alpha*Sk + b[k]*Bk)
            v1_0 = (v1==0)
            ln_v1=np.log(v1)
            ln_v1[v1_0]=0.0
            
            t1=t1+np.sum(Nk * ln_v1) 
            t2=t2+np.sum(alpha*Sk + b[k]*Bk)
            #print(t1,t2, (alpha*Sk + b[k]*Bk))
        res=t1-t2
        return -res




    def calc_bk(self,alpha):
        bk=[]
        for k in range(len(self.background_list)):
            count_map=self.count_list[k]
            exp_map=self.exposure_list[k]
            bkg_map=self.background_list[k]
            Nk=count_map.data *  self.exclusion_list[k].data
            Sk=self.diffuse_list[k].data * exp_map.data * self.exclusion_list[k].data
            Bk=bkg_map.data * self.exclusion_list[k].data  #multiply with exposure?

            not_has_exposure = ~(exp_map.data > 0)
            not_has_bkg = ~(Bk > 0)
            has_bkg=~not_has_bkg
            
            S_B=np.divide(Sk,Bk)
            S_B[not_has_exposure]=0.0
            S_B[not_has_bkg]=0.0

            #Sk is nan for large sep.. to be investigated. temp soln
            if np.isnan(np.sum(S_B)):
                S_B=np.zeros(S_B.shape)


            delta=np.power(np.sum(Nk)/np.sum(Bk),2.0) - 4.0 * alpha * np.sum(S_B)/np.sum(Bk)
            #print(np.sum(Nk),np.sum(Bk),np.sum(S_B))
            if delta<0.0:
                bk=999
                #print("delta is %f for obs no %s",delta,k)
                

            bk1=(np.sum(Nk)/np.sum(Bk) - np.sqrt(delta))/2.0
            bk2=(np.sum(Nk)/np.sum(Bk) + np.sqrt(delta))/2.0
            bk.append(bk2)
        
        return bk



            







    

