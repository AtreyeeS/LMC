import numpy as np
import iminuit
import matplotlib.pyplot as plt


# Define a simple 2D function with minimum
my_func = lambda x,y : (x-xmin)**2 + (y - ymin)**2

xmin =2.34
ymin = np.pi

# Find minimum
minp = iminuit.Minuit(my_func)
minp.migrad()

print(minp.values)



mean = 1.3
stddev = 1.9

# Create array of random variables
x = np.random.normal(mean,stddev,1000)

plt.hist(x,20,normed=True)

# Now a ML fit 
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

mle = loglikelihood(x,gaussian)

m = iminuit.Minuit(mle,mu=0.4,sigma=0.3)
m.migrad()

xplot = np.linspace(x.min(),x.max(),500)
plt.plot(xplot,gaussian(xplot,**m.values),color='r')

plt.show()
