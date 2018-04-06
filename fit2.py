import numpy as np
import iminuit
import matplotlib.pyplot as plt
from sherpa.models import Scale1D, NormGauss1D

mean = 1.3
stddev = 1.9
nevt = 100
# Create array of random variables
x = np.random.normal(mean,stddev,nevt)

plt.hist(x,20,normed=True)


# here we use sherpa models
gaus1 = NormGauss1D('g1')

# one issue is the way the __call__ and calc functions are defined

# gaus1.calc(pars, xlo, *args, **kwargs)
# 1 arg after parameter values list: simple evaluation
print gaus1.calc([10.0,0.0,1.0],0.)
print gaus1.calc([par.val for par in gaus1.pars],0.)
# 2 args after parameter values list: integration between xlo and xhi
print gaus1.calc([par.val for par in gaus1.pars],-np.inf,np.inf)

# gaus1.__call__(*args, **kwargs)
# 1 arg: simple evaluation
print gaus1(0.)
# 2 args : integration between xlo and xhi
print gaus1(-np.inf,np.inf)

# The issue now is due to the fact that models objects don't have func_code or __code__ associated
# therefore iminuit cannot automatically find the parameters
# We have to give them 


class loglikelihood:
    def __init__(self,data1D, model):
        self.data = data1D
        self.model = model

    def __call__(self,*arg):
        for i,val in enumerate(arg):
            self.model.pars[i].val = val
        res = np.log(self.model(self.data))
        return -res.sum()
    # This is a likelihood need an error_def = 0.5
    def default_errordef(self):
        return 0.5

def MakeMinuit(data,model):
    mle = loglikelihood(data,model)

    # I want to pass arguments to iminuit programmatically
    param_names = dict(forced_parameters=[model.name+'_'+_.name for _ in model.pars])
    param_vals = dict([(model.name+'_'+_.name,_.val) for _ in model.pars])
    param_fix = dict([('fix_'+model.name+'_'+_.name,_.frozen) for _ in model.pars])
    param_lims = dict([['limit_'+model.name+'_'+_.name,(np.maximum(_.min,-30),np.minimum(_.max,30))] for _ in model.pars])

    kwargs = dict(param_names,**param_vals)
    kwargs.update(param_fix)
    kwargs.update(param_lims)
    
    return iminuit.Minuit(mle,**kwargs)
    

gaus1.pos=0.
gaus1.fwhm=1.
gaus1.ampl.freeze()

m=MakeMinuit(x,gaus1)
m.migrad()
m.minos()

xplot = np.linspace(x.min(),x.max(),500)
plt.plot(xplot,gaus1(xplot),color='r')

plt.figure()
m.draw_mncontour('g1_fwhm','g1_pos', nsigma=3)

#plt.show()


# now add a second gaussian
mean2 = -0.6
stddev2 = 2.3
nevt2 = 400
# Create array of random variables
y = np.random.normal(mean2,stddev2,nevt2)

# here we use sherpa models
gaus2 = NormGauss1D('g2')

gaus1.pos=1.
gaus1.fwhm=1.
gaus2.pos=0.
gaus2.fwhm=1.

my_model = gaus1 + gaus2
m=MakeMinuit(np.append(x,y),my_model)
#m.migrad()
#m.minos()
