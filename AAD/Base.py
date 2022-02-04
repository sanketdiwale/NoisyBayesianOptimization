import numpy as np
import GPy
import GPyOpt
import os
import signal
import sys
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab
from pathlib import Path 
from GPyOpt.optimization.acquisition_optimizer import ContextManager
from GPyOpt.plotting.plots_bo import plot_convergence #,plot_acquisition
from IPython import embed
import matplotlib.pyplot as plt
# from GPyOpt.util.duplicate_manager import DuplicateManager
class BO:
    def __init__(self,objective_function,kernel_function,gpmodel_function,input_shape=(1,),output_shape=(1,),model_dir:str='out',restart:bool=True,num_init_points=5,noise_var=10.,sparse=False,model_type="gpy_regression"):
        self.objective_function = objective_function
        self.search_space = GPyOpt.Design_space(space=self.objective_function._search_domain)
        self.noise_var = noise_var
        self.acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.search_space) # optimizer used to find the maximum of the acquisition function
        self.context = None
        self.model_dir = model_dir
        Path(os.path.join(self.model_dir,'acq_plots')).mkdir(parents=True,exist_ok=True)
        self.num_init_points = num_init_points
        self.mean_function=None
        self.sparse = sparse
        self.model_type = model_type
        signal.signal(signal.SIGINT, self.signal_handler)
        
        if( restart==False and os.path.isfile(os.path.join(model_dir,'model_params.npy')) \
                and os.path.isfile(os.path.join(model_dir,'X.npy')) and \
                os.path.isfile(os.path.join(model_dir,'y.npy')) and \
                os.path.isfile(os.path.join(model_dir,'acq_params.npy'))):
            self.X = np.load(os.path.join(model_dir,'X.npy'))
            self.y = np.load(os.path.join(model_dir,'y.npy'))
            self.model = gpmodel_function(self.X, self.y,kernel_function, noise_var=self.noise_var,initialize=False)
            self.model.update_model(False) # do not call the underlying expensive algebra on load
            self.model.initialize_parameter() # Initialize the parameters (connect the parameters up)
            self.model[:] = np.load(os.path.join(model_dir,'model_params.npy')) # Load the parameters
            self.model.update_model(True) # Call the algebra only once
            self.acquisition_params = np.load(os.path.join(model_dir,'acq_params.npy'))
        else:
            initial_design = GPyOpt.experiment_design.initial_design('random',self.search_space,self.num_init_points)
            self.X = initial_design
            self.y = self.objective_function.evaluate(self.X)

            # if not self.sparse:
            if self.model_type == "gpy_regression":
                self.model = gpmodel_function(self.X, self.y, kernel=kernel_function, noise_var=self.noise_var, mean_function=self.mean_function)
            else:
                self.model = gpmodel_function
            # else:
            #     self.model = gpmodel_function(self.X, self.y, kernel=kernel_function, num_inducing=self.num_inducing, mean_function=self.mean_function)

            self.exact_feval = True
            if self.model_type == "gpy_regression":
                # --- restrict variance if exact evaluations of the objective
                if self.exact_feval:
                    self.model.Gaussian_noise.constrain_fixed(self.noise_var, warning=False)
                else:
                    # --- We make sure we do not get ridiculously small residual noise variance
                    self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)
            
                # self.model = gpmodel_function(self.X, self.y,kernel_function)
            self.model.analytical_gradient_prediction = True
        self.Y_best = np.empty((0,))
        self.X_best = np.empty((0,self.X.shape[1]))
        if(self.X.shape[1]<=2): 
            self.call_plot = True
        else:
            self.call_plot = False
    
    def save(self):
        np.save(os.path.join(self.model_dir,'X.npy'),self.X)
        np.save(os.path.join(self.model_dir,'y.npy'),self.y)
        np.save(os.path.join(self.model_dir,'model_params.npy'),self.model.param_array)
        np.save(os.path.join(self.model_dir,'acq_params.npy'),self.acquisition_params)
    
    def get_optimum_value(self):
        self._compute_results()
        return self.fx_opt
    def get_optimal_candidate(self):
        self._compute_results()
        return self.x_opt
    def update_model(self):
        if self.model_type == "gpy_regression":
            self.model.set_XY(self.X, self.y)
        else:
            self.model.fit(self.X,self.y,batch_size=min(100,self.X.shape[0]), epochs=500)
        # self.model.optimize()
    def update_Acquisition_params(self):
        pass
    def update_Acquisition_params_post(self):
        pass

    def get_next_candidates(self):
        ## --- Update the context if any
        self.acquisition_function.optimizer.context_manager = ContextManager(self.search_space, self.context)
        # ### --- Activate de_duplication
        # if self.de_duplication:
        #     duplicate_manager = DuplicateManager(space=self.search_space, zipped_X=self.X, pending_zipped_X=None, ignored_zipped_X=None)
        # else:
        #     duplicate_manager = None
        return self.search_space.zip_inputs(self.acq_evaluator.compute_batch(duplicate_manager=None, context_manager= self.acquisition_function.optimizer.context_manager))

    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        # self.Y_best = best_value(self.Y)
        if self.model_type == "gpy_regression":
            ypred,vpred = self.model.predict(self.X, full_cov=False, include_likelihood=False)
        else:
            ypred = self.model(self.X)
            mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
            ypred = mu[:, 0]
            vpred = np.sqrt(beta / (v * (alpha - 1)))
        # embed()
        a = np.argmin(ypred+np.sqrt(vpred)); 
        self.Y_best = np.hstack((self.Y_best,ypred[a]))
        self.x_opt = self.X[a,:]
        self.X_best = np.vstack((self.X_best,self.X[[np.argmin(ypred)],:]))
        self.fx_opt = ypred[a] #np.min(self.Y)

    def add_observations(self,X,Y):
        self.X = np.vstack((self.X,X))
        self.y = np.vstack((self.y,Y))
    def get_objective_values(self,X):
        return self.objective_function.evaluate(X)
    def run_iteration(self):
        '''Runs one iteration of the Bayesian optimization with specified strategy'''
        self.update_model()
        self.update_Acquisition_params()
        r = self.get_next_candidates()
        self.suggesstion = r
        self._compute_results() # update the metrics of the optimizer for record
        if(self.call_plot):
            self.plot_acquisition(os.path.join(self.model_dir,'acq_plots','iter'+str(self.X.shape[0]-self.num_init_points)+'.png'))
        yr = self.get_objective_values(r)
        self.add_observations(r,yr)
        self.update_Acquisition_params_post()
        '''end of the iteration'''
        
    def run(self,max_iters):
        for i in range(max_iters):
            self.run_iteration()
            print('Iteration num',i+1)
        self.save()

    def signal_handler(self,sig, frame):
        print('You pressed Ctrl+C!')
        self.save()
        sys.exit(0)

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
            acqu = -1.*acquisition_function._compute_acq(x_grid)
            acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu))+1e-3)
            m, v = model.predict(x_grid,include_likelihood=True)
            m, v0 = model.predict(x_grid,include_likelihood=False)


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

    def plot_acquisition(self, filename=None, label_x=None, label_y=None):
        """
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        :param label_x: Graph's x-axis label, if None it is renamed to 'x' (1D) or 'X1' (2D)
        :param label_y: Graph's y-axis label, if None it is renamed to 'f(x)' (1D) or 'X2' (2D)
        """
        model_to_plot = self.model
        return self.plot_acquisition_internal(self.acquisition_function.space.get_bounds(),
                                self.X.shape[1],
                                self.model,
                                self.X,
                                self.y,
                                self.acquisition_function,
                                self.suggesstion,
                                filename,
                                label_x,
                                label_y)

    def plot_convergence(self,filename=None):
        """
        Makes twp plots to evaluate the convergence of the model:
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        return plot_convergence(self.X,self.Y_best.tolist(),filename)

    def compute_regret(self,true_opt,best_Y):
        return (best_Y-true_opt).T

    def plot_regret(self,regret_array,run,filename=None):
        plt.figure()
        regret_array = regret_array/(3*self.objective_function.f_opt)
        x_min = np.min(regret_array[0:run+1,:],axis=0)
        x_max = np.max(regret_array[0:run+1,:],axis=0)
        x_mean = np.mean(regret_array[0:run+1,:],axis=0)
        plt.plot(x_min,label="min")
        plt.plot(x_mean,label="average")
        plt.plot(x_max,label="max")
        plt.gca().set_xscale('linear')
        plt.gca().set_yscale('log')
        h = plt.ylabel(r'$\frac{\Delta f_{opt}}{3\sigma_{\eta}}$',rotation=0,fontsize=16)
        # h.set_rotation(0); 
        plt.gca().set_xlabel(r'$iterations$')
        plt.gca().yaxis.set_label_coords(-0.12,0.4)
        # plt.gca().set_yscale('log')
        plt.grid(True)
        # plt.grid(True)
        plt.legend(loc='upper right')
        if filename!=None:
            plt.savefig(filename)

if __name__=="__main__":
    from AAD.Objectives.ObjFunc import IndTimeModel
    objective = IndTimeModel(problemID="QU_DI")
    Optimizer = BO(objective_function=objective,kernel_function=GPy.kern.RBF(1),gpmodel_function=GPy.models.GPRegression)
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        Optimizer.save()
        sys.exit(0)
    max_iters = 10
    Optimizer.run(max_iters=max_iters)
    