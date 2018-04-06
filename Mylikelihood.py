class MyLikelihood(object):
    """
    Class to compute the b_k and do the compute the likelihood function
    """

    def __init__(self,image_list,diffuse_map,exclusion_mask=None):
        self.count_list=[row[0] for row in image_list]
        self.exposure_list=[row[1] for row in image_list]
        self.background_list=[row[2] for row in image_list]
        self.exclusion_mask=exclusion_mask
        self.diffuse_map=diffuse_map



    def __call__(self,alpha):
        result=self.calc_res(alpha)
        return result
        

    def calc_res(self,alpha):
        #calculates the log likelihood function to minimise 
        b = self.calc_bk(akpha)
        res=0.0
        for k in range(len(b)):
            count_map=self.count_list[k]
            exp_map=self.exposure_list[k]
            bkg_map=self.background_list[k]
            Nk=count_map.data * exp_map.data * self.exclusion_mask.data
            Sk=self.diffuse_map.data * exp_map.data * self.exclusion_mask.data
            Bk=bkg_map.data * self.exclusion_mask.data * exp_map.data #multiply with exposure?

            tk=np.sum(Nk * np.log(alpha*Sk + b[k]*Bk)) - np.sum(alpha*Sk + b[k]*Bk)
            res=res+tk
        return res




    def calc_bk(self,alpha):
        bk1=[]
        bk2=[]
        for k in range(len(self.background_list)):
            count_map=self.count_list[k]
            exp_map=self.exposure_list[k]
            bkg_map=self.background_list[k]
            Nk=count_map.data * exp_map.data * self.exclusion_mask.data
            Sk=self.diffuse_map.data * exp_map.data * self.exclusion_mask.data
            Bk=bkg_map.data * self.exclusion_mask.data * exp_map.data #multiply with exposure?

            not_has_exposure = ~(exp_map.data > 0)
            not_has_bkg = ~(bkg_map.data > 0)
            S_B=Sk/Bk
            S_B[not_has_exposure]=0.0
            S_B[not_has_bkg]=0.0

            delta=np.power(np.sum(Nk)/np.sum(Bk),2.0) - 4.0 * alpha * np.sum(S_B)/np.sum(Bk)
            if delta<0.0:
                bk=999
                continue

            bk1=(np.sum(Nk)/np.sum(Bk) - np.sqrt(delta))/2.0
            bk2=(np.sum(Nk)/np.sum(Bk) + np.sqrt(delta))/2.0
            bk2.append(bk2)
            bk1.append(bk1)
        
        return bk2



            







    

