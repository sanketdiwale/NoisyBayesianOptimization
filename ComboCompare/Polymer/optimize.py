import numpy as np
import cPickle as pickle
import scipy
import combo
from Objectives.ObjFunc import IndTimeModel
from IPython import embed
import os, sys
# from pyDOE import lhs
# objective_model = IndTimeModel(problemID="QU_GR",noisy=True)
# from scipy.stats.qmc import LatinHypercube

if len(sys.argv)!=2:
    sys.exit("Script needs an integer argument to be used as the batch index.")
else:
    batch = int(sys.argv[1])
    print("Running batch "+str(batch))

class simulator:
    def __init__(self):
        self.objective_model = IndTimeModel(problemID="QU_GR",noisy=True)
        b = self.objective_model.bounds
        Ngrid = 25
        x = []; lengthscales = []; m= []; n = []
        for i in range(b.shape[0]):
            x.append(np.linspace(b[i][0],b[i][1],Ngrid))
            lengthscales.append(abs(b[i][1]-b[i][0])/10.)
            m.append(max(b[i][0],b[i][1]))
            n.append(min(b[i][0],b[i][1]))
        self.lengthscales = np.array(lengthscales)
        r = np.meshgrid(*x)
        action_size = Ngrid**b.shape[0]
        X = np.empty((action_size,b.shape[0]))
        for i in range(b.shape[0]):
            X[:,[i]] = r[i].reshape(-1,1)

        # engine = LatinHypercube(d=b.shape[0])
        # sample = engine.random(n=action_size)
        # embed()
        # sample = np.array(n).reshape((1,b.shape[0]))+np.dot(np.random.rand(action_size,b.shape[0]),np.diag(np.array(m)-np.array(n)))# lhs(action_size, [samples, criterion, iterations])
        self.X = X        
        # self.X = sample
    def getsamples(self,Ngrid = 25):
        b = self.objective_model.bounds
        m= []; n = []; action_size = Ngrid**b.shape[0]
        for i in range(b.shape[0]):
            m.append(max(b[i][0],b[i][1]))
            n.append(min(b[i][0],b[i][1]))
        return np.array(n).reshape((1,b.shape[0]))+np.dot(np.random.rand(action_size,b.shape[0]),np.diag(np.array(m)-np.array(n)))

    def __call__(self,action):
        r = self.X[action,:]
        # embed()
        x = self.objective_model.evaluate(r)
        return -x[0][0]

    def compute_regret(self,action_best,policy):
        # embed()
        x = policy.test.X[action_best.astype(int).tolist(),:]
        y = self.objective_model.evaluate_true(x)
        return y - self.objective_model.f_opt

sim = simulator()
X = sim.X #combo.misc.centering( sim.X )

model = combo.gp.core.model(cov = combo.gp.cov.gauss( num_dim = None, ard = False ), mean = combo.gp.mean.const(), lik = combo.gp.lik.gauss())
# params are taken by combo as [noise_std_dev,prior_mean,kernel_var_scale,kernel_len_scale_inv] for the sq. exp kernel (kernel_var_scale^2)*e^{-0.5*(x-y)^2*kernel_len_scale_inv^2}
params = np.array([1,0,1,np.min(sim.lengthscales)**(-1)])
model.set_params(params)
predictor = combo.gp.predictor(config=None, model = model)
policy = combo.search.discrete.policy(test_X=X)
# policy.set_seed( 0 )

res = policy.random_search(max_num_probes=1, simulator=sim)

# embed()
# res = policy.bayes_search(max_num_probes=200, simulator=sim, score='EI', 
#                                                   interval=10,num_search_each_probe=1,num_rand_basis=5000) #predictor=predictor
# embed()
Bs= 2000/10
x_comp = np.empty((0,X.shape[1]))
regrets = np.empty((0,1))
for i in range(Bs):
    # policy = combo.search.discrete.policy(test_X=sim.getsamples())
    policy.test = policy._set_test(np.vstack((x_comp,sim.getsamples())))
    res = policy.bayes_search(max_num_probes=10, simulator=sim, score='EI', 
                                interval=-1,num_search_each_probe=1, predictor=predictor)
    best_fx, best_action = res.export_all_sequence_best_fx()
    x_comp = policy.test.X #[best_action[-1].astype(int).tolist(),:]
    # regrets = np.vstack((regrets,sim.compute_regret(best_action,policy)))
    # embed()

print 'f(x)='
print -res.fx[0:res.total_num_search]
best_fx, best_action = res.export_all_sequence_best_fx()
print 'current best'
print -best_fx
print 'current best action='
print best_action
print 'history of chosed actions='
print res.chosed_actions[0:res.total_num_search]

# embed()
basedir = os.path.join('results_GP12','batch'+str(batch))
if not os.path.exists(basedir):
    os.makedirs(basedir)
regrets = sim.compute_regret(best_action,policy)
np.save(os.path.join(basedir,'com_regrets.npy'),regrets)
res.save(os.path.join(basedir,'results.npz'))
del res

# load the results
# res = combo.search.discrete.results.history()
# res.load('test.npz')


# embed()
