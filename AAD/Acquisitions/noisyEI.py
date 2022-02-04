from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.EI import AcquisitionEI
import GPyOpt
import GPy
from IPython import embed
import numpy as np
from scipy import stats

class Aug_EI(AcquisitionBase):
    analytical_gradient_prediction = True
    def __init__(self, model, space, kernel, acquisition_base,power=1,optimizer=None, cost_withGradients=None):
        super(Aug_EI, self).__init__(model, space, optimizer,cost_withGradients=cost_withGradients)
        self.acq_base = acquisition_base
        self.model = model
        self.kernel = kernel
        self.name = 'aug_'+self.acq_base.name
        self.power = power
    
    def _compute_acq(self,x):
        a = self.acq_base._compute_acq(x)
        # embed()
        _,v = self.model.predict(x,full_cov=False, include_likelihood=False)
        aug = 1. - self.model.noise_var/(self.model.noise_var+v)
        return a*(aug**self.power)

    def _compute_acq_withGradients(self,x):
        a,g = self.acq_base._compute_acq_withGradients(x)

        _,v = self.model.predict(x,full_cov=False, include_likelihood=False)
        aug = 1. - self.model.noise_var/(self.model.noise_var+v)
        _,dv_dx = self.model.predictive_gradients(x)
        g = g*(aug**self.power) - self.power*dv_dx*(aug**(self.power-1))*a*self.model.noise_var/(self.model.noise_var+v)**2
        
        return a*(aug**self.power) , g

class data_EI(AcquisitionBase):
    analytical_gradient_prediction = True
    def __init__(self, model, space, kernel, acquisition_base,optimizer=None, cost_withGradients=None, num_samples= 15):
        super(data_EI, self).__init__(model, space, optimizer,cost_withGradients=cost_withGradients)
        self.num_samples = num_samples
        self.acq_base = acquisition_base
        self.model = model
        self.kernel = kernel
        self.name = 'data_EI_'+self.acq_base.name

    def acquisition_function(self,x):
        a = self.acq_base.acquisition_function(x)
        # embed()
        k = self.kernel.K(x,self.model.model.X)
        ak = np.atleast_2d(np.sum(k,axis=1)).T #/k.shape[1]
        a = a-(ak+1e-3)**(-1)
        return a

    def acquisition_function_withGradients(self,x):
        a,g = self.acq_base.acquisition_function_withGradients(x)
        k = self.kernel.K(x,self.model.model.X)
        ak = np.atleast_2d(np.sum(k,axis=1)).T #/k.shape[1]
        a = a-(ak+0.0001)**(-1)
        # embed()
        dk_dx = self.kernel.gradients_X(np.eye(self.model.model.X.shape[0]),x,self.model.model.X) #/k.shape[1] 
        # dk_dx = np.sum()
        g = g - np.dot(np.diag(((ak+0.0001)**(-2))),dk_dx)
        return a,g
class noisy_EI_exp(AcquisitionBase):
    analytical_gradient_prediction = True
    
    def __init__(self, model, space, kernel, optimizer=None, cost_withGradients=None, par_a=1, par_b=1, num_samples= 15):
        super(noisy_EI_exp, self).__init__(model, space, optimizer,cost_withGradients=cost_withGradients)
        
        self.par_a = par_a
        self.par_b = par_b
        self.num_samples = num_samples
        # self.x_dim = len(space.config_space)
        # self.model = model
        # self.samples = np.random.uniform(0,1,(self.num_samples,self.x_dim))#beta(self.par_a,self.par_b,self.num_samples)
        self.EI = AcquisitionEI(model, space, optimizer, cost_withGradients)
        self.models = []
        self.EIs = []
        for k in range(self.num_samples):
            self.models.append( GPyOpt.models.GPModel(kernel=kernel,max_iters=0,noise_var=1.) ) #,max_iters=10
            self.EIs.append( AcquisitionEI(self.models[k], space, optimizer, cost_withGradients) )
        self.name = 'noisyEI_exp'

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return noisy_EI(model, space, optimizer, cost_withGradients, jitter=config['jitter'],num_samples=config['num_samples'])
    
    def acquisition_function(self,x):
        # acqu_x = np.zeros((x.shape[0],1))    
        # embed()   
        m, _ = self.model.predict(self.model.model.X)
        isMCMC = False 
        # use the exponential model for the noise
        if isinstance(m,list):
            m = np.array(m)
            self.samples = np.log(np.random.exponential(np.exp(m),(self.num_samples,)+m.shape))[:,:,:,0] # arising from MCMC model
            isMCMC = True
            # embed()
        else:
            self.samples = np.log(np.random.exponential(np.exp(m),(self.num_samples,)+m.shape))[:,:,0]
        for k in range(self.num_samples):
            if isMCMC:
                f_k = np.reshape(self.samples[[k],:],(m.shape[0],-1,1))
                for j in range(f_k.shape[0]):
                    self.models[k].updateModel(self.model.model.X,f_k[j,:,:],None,None)
            else:
                f_k = np.reshape(self.samples[[k],:],(-1,1))
                self.models[k].updateModel(self.model.model.X,f_k,None,None)
        # embed()
        nei = 0.
        for k in range(self.num_samples):
            nei = nei + self.EIs[k].acquisition_function(x)
        return nei/self.num_samples
    
    def acquisition_function_withGradients(self,x):
        # embed()
        m, _ = self.model.predict(self.model.model.X)#,True,True) # predict full covariance, with additive noise included
        # A = np.linalg.cholesky(S)
        # self.samples = np.random.uniform(0,1,(self.num_samples,self.model.model.X.shape[0]))
        # q = stats.norm.ppf(self.samples)
        self.samples = np.log(np.random.exponential(np.exp(m),(self.num_samples,)+m.shape))[:,:,0]
        for k in range(self.num_samples):
            # draw random fake observations from the GP posterior
            f_k = np.reshape(self.samples[[k],:],(-1,1)) #np.dot(A,q[[k],:].T)+m # This needs to be fixed
            self.models[k].updateModel(self.model.model.X,f_k,None,None)
        acqu_x      = np.zeros((x.shape[0],1))       
        acqu_x_grad = np.zeros(x.shape)
        
        for k in range(self.num_samples):
            # self.EI.jitter = self.samples[k]       
            acqu_x_sample, acqu_x_grad_sample =self.EIs[k].acquisition_function_withGradients(x) 
            acqu_x += acqu_x_sample
            acqu_x_grad += acqu_x_grad_sample           
        return acqu_x/self.num_samples, acqu_x_grad/self.num_samples

class noisy_EI(AcquisitionBase):
    
    analytical_gradient_prediction = True
    
    def __init__(self, model, space, kernel, optimizer=None, cost_withGradients=None, par_a=1, par_b=1, num_samples= 15):
        super(noisy_EI, self).__init__(model, space, optimizer,cost_withGradients=cost_withGradients)
        
        self.par_a = par_a
        self.par_b = par_b
        self.num_samples = num_samples
        # self.x_dim = len(space.config_space)
        # self.model = model
        # self.samples = np.random.uniform(0,1,(self.num_samples,self.x_dim))#beta(self.par_a,self.par_b,self.num_samples)
        self.EI = AcquisitionEI(model, space, optimizer, cost_withGradients)
        self.models = []
        self.EIs = []
        for k in range(self.num_samples):
            self.models.append( GPyOpt.models.GPModel(kernel=kernel,max_iters=0,noise_var=0.0001) )
            self.EIs.append( AcquisitionEI(self.models[k], space, optimizer, cost_withGradients) )
        self.name = 'noisyEI'

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return noisy_EI(model, space, optimizer, cost_withGradients, jitter=config['jitter'],num_samples=config['num_samples'])
    
    def acquisition_function(self,x):
        # acqu_x = np.zeros((x.shape[0],1))    
        # embed()   
        m, S = self.model._predict(self.model.model.X,True,True) # predict full covariance, with additive noise included
        A = np.linalg.cholesky(S)
        self.samples = np.random.uniform(0,1,(self.num_samples,self.model.model.X.shape[0]))
        # draw random fake observations from the GP posterior
        q = stats.norm.ppf(self.samples)
        for k in range(self.num_samples):
            f_k = np.dot(A,q[[k],:].T)+m # This needs to be fixed
            self.models[k].updateModel(self.model.model.X,f_k,None,None)
        # embed()
        nei = 0.
        for k in range(self.num_samples):
            nei = nei + self.EIs[k].acquisition_function(x)
        return nei/self.num_samples
    
    def acquisition_function_withGradients(self,x):
        # embed()
        m, S = self.model._predict(self.model.model.X,True,True) # predict full covariance, with additive noise included
        A = np.linalg.cholesky(S)
        self.samples = np.random.uniform(0,1,(self.num_samples,self.model.model.X.shape[0]))
        q = stats.norm.ppf(self.samples)
        for k in range(self.num_samples):
            # draw random fake observations from the GP posterior
            f_k = np.dot(A,q[[k],:].T)+m # This needs to be fixed
            self.models[k].updateModel(self.model.model.X,f_k,None,None)
        acqu_x      = np.zeros((x.shape[0],1))       
        acqu_x_grad = np.zeros(x.shape)
        
        for k in range(self.num_samples):
            # self.EI.jitter = self.samples[k]       
            acqu_x_sample, acqu_x_grad_sample =self.EIs[k].acquisition_function_withGradients(x) 
            acqu_x += acqu_x_sample
            acqu_x_grad += acqu_x_grad_sample           
        return acqu_x/self.num_samples, acqu_x_grad/self.num_samples