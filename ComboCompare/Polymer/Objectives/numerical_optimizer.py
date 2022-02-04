# import a optimization solver and name it optim_solver (the script below then doesn't need to change much)
import numpy as np
from scipy.optimize import minimize as optim_solver
from scipy.optimize import Bounds
from AAD.Objectives.ObjFunc import IndTimeModel
from IPython import embed
# call the optimizer
Model = IndTimeModel(problemID="QU_GR",noisy=True)

objective_function = lambda x: Model.evaluate_true_log(x).ravel()
lb = Model.bounds[:,0]; ub = Model.bounds[:,1]; 
bounds = Bounds(lb,ub); 
sampl = np.random.uniform(low=lb, high=ub, size=(5,)+lb.shape)
x0 = np.array([1.1,0.32,0.56,1.2]) #Model.x_opt_true # give an initial guess for the solution
sampl = np.vstack( (x0,sampl) )
sols = []
fopt = np.inf
xopt = None
for n in range(sampl.shape[0]):
    x0 = sampl[n] # give an initial guess for the solution
    sols.append(optim_solver( objective_function , x0 , bounds = bounds ))
    if(sols[-1]['success'] and sols[-1]['fun'][0]<=fopt):
        xopt = sols[-1]['x']
        fopt = sols[-1]['fun'][0]

print('optimal solution: ',xopt)
print('optimal value: ',fopt)
embed()