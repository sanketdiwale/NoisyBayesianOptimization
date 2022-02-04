from numpy.core.numeric import NaN
from multiprocessing import Pool

from numpy.lib.function_base import vectorize
from AAD.Base import BO
import GPy
import GPyOpt
import numpy as np
from GPyOpt.acquisitions.LCB import AcquisitionLCB
from GPyOpt.acquisitions.EI import AcquisitionEI
from AAD.Acquisitions.noisyEI import Aug_EI
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab
from IPython import embed
import math
import sys, os
from AAD.Strategies.AGPLCB.AGPLCB import AGPLCB
from AAD.Acquisitions.BoostedAcquisition import BoostedAcquisition
import emcee
from pathlib import Path 
class BoostedGPLCB(AGPLCB):
    analytical_gradient_prediction = True
    def __init__(self,objective_function,kernel_function,gpmodel_function,input_shape=(1,),output_shape=(1,),model_dir:str='out',restart:bool=True,num_init_points=1,noise_var=0.01,sparse=False,parent_opt=None):
        amodel_dir = model_dir
        if(parent_opt is not None):
            amodel_dir = os.path.join(parent_opt.model_dir,model_dir)
        super(BoostedGPLCB,self).__init__(objective_function,kernel_function,gpmodel_function,input_shape,output_shape,amodel_dir,restart,num_init_points,noise_var,sparse)
        self.parent_opt = parent_opt
        self.noise_var = noise_var
        acquisition_function0 = AcquisitionEI(self,self.search_space,self.acquisition_optimizer,jitter = 0.5)
        if(self.parent_opt is not None):
            acquisition_function0.fmin = self.parent_opt.fx_opt
        else:
            acquisition_function0.fmin = 1000.
        acquisition_function = Aug_EI(self, self.search_space, kernel_function, acquisition_function0,power=0,optimizer=self.acquisition_optimizer)
        if(self.parent_opt is not None):
            self.acquisition_function = BoostedAcquisition(acquisition_function,self.parent_opt)
        else:
            self.acquisition_function = BoostedAcquisition(acquisition_function)
        self.fx_opt = 0.
        self.x_opt = None
        if(self.parent_opt is not None):
            self.acquisition_function.parent_acquisition.fmin = self.parent_opt.fx_opt
            
        self.acq_evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition_function)
        # MC sampler
        self.nwalkers = 32
        ndim = len(objective._search_domain)
        if(parent_opt is None):
            p0 = np.random.uniform(low=self.objective_function.bounds[:,0],high=self.objective_function.bounds[:,1],size=(self.nwalkers,ndim))
        else:
            p0 = self.parent_opt.mc_state
        # p0 = np.random.rand(self.nwalkers, ndim)
        # self.pool = Pool()
        self.mc_sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_Acq,vectorize=True) #, args=[means, cov])
        self.acquisition_params = np.empty((0,ndim+1))
        # run initial burn in
        self.mc_state = self.mc_sampler.run_mcmc(p0, 100,progress=True)
        self.mc_sampler.reset()
        if(parent_opt is None):
            self.run_mc = 1000
        else:
            self.run_mc = 100
        print('Initialized optimizer')
        self.q = 10

    def update_Acquisition_params(self):
        self.acquisition_function.fmin = self.fx_opt
        if self.x_opt is not None:
            r = np.hstack((self.fx_opt,self.x_opt))
            self.acquisition_params = np.vstack( (self.acquisition_params, r) )
        # try:
        self.acquisition_function.acquisition.fmin = self.fx_opt
        # except:
        #     embed()
        # if(self.parent_opt is not None):
        #     self.acquisition_function.parent_acquisition.fmin = self.fx_opt

    def log_Acq(self,x):
        # embed()
        acqu = self.acquisition_function._compute_acq(np.atleast_2d(x)) #+ 1000.
        # embed()
        acqu[np.any(x<self.objective_function.bounds[:,0])] = 0.
        acqu[np.any(x>self.objective_function.bounds[:,1])] = 0.
        # acqu_normalized = (-acqu - np.min(-acqu,axis=1))/(np.max(-acqu - np.min(-acqu,axis=1),axis=1)+1e-3)
        m = np.log(acqu)
        if np.isnan(np.sum(m)):
            m[np.isnan(m)]= -np.inf
            # embed()
        return m

    def add_observations(self,X,Y):
        if self.parent_opt is not None:
            mpred,vpred = self.parent_opt.predict(X)
            Y = (Y - mpred)
        
        self.X = np.vstack((self.X,X))
        self.y = np.vstack((self.y,Y))

    def predict(self,x,full_cov=False, include_likelihood=False):
        ypred,vpred = self.model.predict(x, full_cov=full_cov, include_likelihood=include_likelihood)
        if self.parent_opt is not None:
            ypred1,vpred1 = self.parent_opt.predict(x, full_cov=full_cov, include_likelihood=include_likelihood)
            ypred = ypred+ypred1
            vpred = vpred + vpred1
        return ypred,vpred

    def get_next_candidates(self):
        self.mc_sampler.reset()
        self.mc_state = self.mc_sampler.run_mcmc(self.mc_state, self.run_mc,progress=True);
        self.run_mc = 100
        samples = self.mc_sampler.get_chain(flat=True)
        k = self.log_Acq(samples)
        samples1 = samples[k[:,0]> -np.inf,:]
        # embed()
        r = samples1[[np.random.randint(samples1.shape[0],size=self.q)],:][0,:,:]
        return r

    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        # self.Y_best = best_value(self.Y)
        X = self.X
        ypred,vpred = self.predict(X)
        
        a = np.argmin(ypred); 
        self.Y_best = np.hstack((self.Y_best,ypred[a]))
        self.x_opt = X[a,:]
        self.X_best = np.vstack((self.X_best,X[[np.argmin(ypred)],:]))
        self.fx_opt = ypred[a] #np.min(self.Y)

    def plot_acquisition_internal(self,bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample,
                     filename=None, label_x=None, label_y=None, color_by_step=True):
        '''
        Plots of the model and the acquisition function in 1D and 2D examples.
        '''
        # Plots in dimension 1
        if input_dim ==1:
            if not label_x:
                label_x = 'x'

            if not label_y:
                label_y = 'f(x)'

            x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
            x_grid = x_grid.reshape(len(x_grid),1)
            acqu = self.acquisition_function._compute_acq(x_grid)
            acqu_normalized = (acqu - min(acqu))/(max(acqu - min(acqu))+1e-3)
            m, v = self.predict(x_grid,include_likelihood=True)
            m, v0 = self.predict(x_grid,include_likelihood=False)
            if(self.parent_opt is not None):
                acqu_p = self.acquisition_function.parent_acquisition._compute_acq(x_grid)
                acqu_p_normalized = (acqu_p - min(acqu_p))/(max(acqu_p - min(acqu_p))+1e-3)

                acqu_c = self.acquisition_function.acquisition._compute_acq(x_grid)
                acqu_c_normalized = (acqu_c - min(acqu_c))/(max(acqu_c - min(acqu_c))+1e-3)
                # embed()


            model.plot_density(bounds[0], alpha=.5)

            plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
            plt.plot(x_grid, m-1.96*np.sqrt(v), 'r-', alpha = 0.2)
            plt.plot(x_grid, m+1.96*np.sqrt(v), 'r-', alpha=0.2)

            plt.plot(x_grid, m-1.96*np.sqrt(v0), 'k--', alpha = 0.1)
            plt.plot(x_grid, m+1.96*np.sqrt(v0), 'k--', alpha=0.1)

            plt.plot(Xdata, Ydata, 'r.', markersize=10)
            plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
            factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))
            ylim_min = min(min(m-1.96*np.sqrt(v))-0.25*factor,min(Ydata)-1.)
            ylim_max = max(max(m+1.96*np.sqrt(v))+0.05*factor,max(Ydata)+1.)
            plt.plot(x_grid,0.2*factor*acqu_normalized+ylim_min, 'r-',lw=2,label ='Acquisition (arbitrary units)')
            if(self.parent_opt is not None):
                plt.plot(x_grid,0.2*factor*acqu_p_normalized+ylim_min, 'g--',lw=2,label ='Parent Acquisition')
                plt.plot(x_grid,0.2*factor*acqu_c_normalized+ylim_min, 'b--',lw=2,label ='Current Acquisition')
            
            # print('Acquisition min: ',min(0.2*factor*acqu_normalized+ylim_min))
            # print('Acquisition max: ',max(0.2*factor*acqu_normalized+ylim_min))
            # embed()
            plt.xlabel(label_x)
            plt.ylabel(label_y)
            plt.ylim(ylim_min,  ylim_max)
            plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
            plt.legend(loc='upper left')


            if filename!=None:
                savefig(filename)
            else:
                plt.show()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

if __name__=="__main__":
    batch = 0
    if len(sys.argv)!=2:
        sys.exit("Script needs an integer argument to be used as the batch index.")
    else:
        batch = int(sys.argv[1])
        print("Running batch "+str(batch))
    from AAD.Objectives.ObjFunc import IndTimeModel
    
    objective = IndTimeModel(problemID="QU_GR",noisy=True)
    bounds = objective._search_domain
    lengthscales = [abs(x['domain'][1]-x['domain'][0])/10. for x in bounds]
    TREE_DEPTH = 1
    opts = []
    max_iters = 200
    com_regrets = np.empty((1,0));
    if objective.noisy:
        noisevar = 1.
    else:
        noisevar = 0.01
    for d in range(TREE_DEPTH):
        if d==0:
            kern = GPy.kern.RBF(len(bounds), variance=1.0,lengthscale=lengthscales, ARD=True)
            kern.lengthscale.constrain_bounded(np.array(lengthscales).min()*0.95,np.array(lengthscales).max()*1.05)
            # kern += GPy.kern.Poly(len(lengthscales), variance=1., scale=1., bias=1., order=3., active_dims=None, name='poly')
            opts.append(BoostedGPLCB(model_dir=os.path.join('test5',objective.name(),'batch'+str(batch)),objective_function=objective,kernel_function=kern,gpmodel_function=GPy.models.GPRegression,noise_var=noisevar,restart=True,num_init_points=1))
            opts[d].lengthscale0= lengthscales
            # opts[d].model.likelihood.variance.constrain_fixed(noisevar)
        else:
            opts.append(BoostedGPLCB(objective_function=objective,kernel_function=GPy.kern.RBF(len(bounds), variance=1.0,lengthscale= opts[d-1].model.kern.lengthscale, ARD=True),gpmodel_function=GPy.models.GPRegression,noise_var=noisevar,restart=True,parent_opt=opts[d-1]))
            opts[d].lengthscale0 = opts[d-1].model.kern.lengthscale
            Path(os.path.join(opts[d].model_dir,'acq_plots')).mkdir(parents=True,exist_ok=True)
        opts[d].run(max_iters=max_iters)
        regrets = opts[d].compute_regret(objective.f_opt,objective.evaluate_true(opts[d].X_best))
        np.save(os.path.join(opts[d].model_dir,'regrets.npy'),regrets)
        com_regrets = np.hstack((com_regrets,regrets))
        opts[d].plot_regret(regrets,regrets.shape[0],os.path.join(opts[d].model_dir,'regret%.03i.png'%(opts[d].X.shape[0])))
        # opts[d].pool.close()
    np.save(os.path.join(opts[0].model_dir,'com_regrets.npy'),com_regrets)
    embed()
