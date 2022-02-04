from AAD.Base import BO
import GPy
import GPyOpt
import numpy as np
from GPyOpt.acquisitions.LCB import AcquisitionLCB
from IPython import embed
import math
import sys, os
class AGPLCB(BO):
    def __init__(self,objective_function,kernel_function,gpmodel_function,input_shape=(1,),output_shape=(1,),model_dir:str='out',restart:bool=True,num_init_points=5,noise_var=0.01,sparse=False,model_type="gpy_regression"):
        super(AGPLCB,self).__init__(objective_function,kernel_function,gpmodel_function,input_shape,output_shape,model_dir,restart,num_init_points,noise_var,sparse,model_type)
        self.acquisition_function = AcquisitionLCB(self.model,self.search_space,self.acquisition_optimizer)
        self.acq_evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition_function)
        self.acq_param_names = ['iter_num','lambda']
        self.regret_estimate = 0
        self.B0 = 1.
        self.delta = 0.01
        self.e = np.log(1./self.delta)
        self.acquisition_params = []

    def ref_regret(self,iter_num):
        return iter_num**0.8

    def update_Acquisition_params_post(self):
        iter_num = self.X.shape[0]
        _,vpred = self.model.predict(self.suggesstion, full_cov=False, include_likelihood=False)
        self.h_t = np.log(1+iter_num)
        I_t = 0.5*np.log(np.linalg.det(self.model.kern.K(self.X))) - 0.5*np.log(self.model.likelihood.variance)
        self.c_t = 4*self.model.likelihood.variance*np.sqrt(I_t+self.e)
        self.beta_t = self.h_t*self.B0 + self.c_t
        self.acquisition_function.exploration_weight = 2.*self.beta_t
        d = self.X.shape[1]; i = iter_num
        gamma = ((i+1)**(d*(d+1)/(5+d*(d+1))))*np.log((i+1)) # assumes matern 5/2 kernel
        self.acquisition_function.exploration_weight  = np.sqrt(4 + 300*gamma*(np.log((i+1)/0.01))**3)
        # embed()
        # self.model.kern.lengthscale= self.lengthscale0/(self.h_t**(1./self.X.shape[1]));
    

if __name__=="__main__":

    # if len(sys.argv)!=2:
    #     sys.exit("Script needs an integer argument to be used as the batch index.")
    # else:
    #     batch = int(sys.argv[1])
    #     print("Running batch "+str(batch))
    from AAD.Objectives.ObjFunc import IndTimeModel
    objective = IndTimeModel(problemID="QU_DI1d")
    bounds = objective._search_domain
    lengthscales = [abs(x['domain'][1]-x['domain'][0])/10. for x in bounds]
    kernel = GPy.kern.RBF(len(bounds), variance=1.0,lengthscale=lengthscales, ARD=True)
    Optimizer = AGPLCB(objective_function=objective,kernel_function=kernel,gpmodel_function=GPy.models.GPRegression,noise_var=10.,restart=False)
    Optimizer.lengthscale0= lengthscales
    max_iters = 30
    Optimizer.run(max_iters=max_iters)
    regrets = Optimizer.compute_regret(objective.f_opt,objective.evaluate_true(Optimizer.X_best))
    Optimizer.plot_regret(regrets,regrets.shape[0],os.path.join(Optimizer.model_dir,'regret%.03i.png'%(Optimizer.X.shape[0])))
    
    embed()