'''
Created on Feb 11, 2021

@author: Sanket
'''

from __future__ import division
# from builtins import object
import numpy as np
import math
import sys
# from IPython import embed

import QuadraticBourque as QU

class IndTimeModel():#object):
    def __init__(self, problemID, noisy=True):
        self.noisy = noisy
        if problemID == "QU_DI1d":
            self._dim = 1
            bounds = [
                {'name': '$\epsilon_{AD}$', 'type': 'continuous', 'domain': (0.6,1.0)}]
            self.f_opt = np.exp(1.391)
            self.x_opt_true = [0.9843]
        elif problemID == "QU_DI":
            self._dim = 3
            bounds = [
                {'name': '$\epsilon_{AD}$', 'type': 'continuous', 'domain': (0.6,1.0)},
                {'name': '$\lambda_{SW}$', 'type': 'continuous', 'domain': (0.9,1.3)},
                {'name': '$\sigma_{SW}$', 'type': 'continuous', 'domain': (0.8,0.95)}]  # problem constraints
            self.f_opt = np.exp(1.391)
            self.x_opt_true = [0.9843,0.9,0.8696]
        elif problemID == "QU_GR1d":
            self._dim = 1
            bounds = [
                {'name': '$\lambda_{SW}$', 'type': 'continuous', 'domain': (0.6,1.4)}]  # problem constraints
            self.f_opt = np.exp(1.994)
            self.x_opt_true = [0.6]
        else:
            self._dim = 4
            bounds = [
                {'name': '$\sigma_{SW}$', 'type': 'continuous', 'domain': (1.05,1.33)},
                {'name': '$\epsilon_{SW}$', 'type': 'continuous', 'domain': (0.7,1.1)},
                {'name': '$\lambda_{SW}$', 'type': 'continuous', 'domain': (0.6,1.4)},
                {'name': '$\epsilon_{AD}$', 'type': 'continuous', 'domain': (0.8,1.2)},
                ]  # problem constraints
            self.f_opt = np.exp(1.994)
            self.x_opt_true = [1.05,1.1,0.6,1.115]

        self._search_domain = bounds
        self._sample_var = 1.0
        self._min_value = 0.0
        self._observations = []
        self._num_fidelity = 0
        self._problemID = problemID
        self.bounds = np.array([x['domain'] for x in self._search_domain])
        self.rng = np.random#.default_rng()
        # embed()

    def evaluate_true(self, x):
        return np.exp(self.evaluate_true_log(x))

    def evaluate_true_log(self,x):
        if self._problemID == "QU_DI1d":
            results = QU.Quadratic_diamond( np.hstack( (x,np.array([[0.9,0.8696]]*x.shape[0])) ) )
            return results
        elif self._problemID == "QU_DI":
            results = QU.Quadratic_diamond(x) 
            # results = results
            return results    
        elif self._problemID == "QU_GR1d":
            results = QU.Quadratic_graphene(np.hstack( (np.array([[1.05,1.1]]*x.shape[0]) ,x, np.array([[1.115]]*x.shape[0]) ) )) #  [1.05,1.1,0.6,1.115]
            # results contain log(\tau_mean)
            return results 
        elif self._problemID == "QU_GR":
            results = QU.Quadratic_graphene(x)
            # results contain log(\tau_mean)
            return results
        else:
            print(self._problemID + str(" is not implemented!"))
            sys.exit(0)

    def evaluate(self, x):
        t = self.evaluate_true(x) # t is tau_mean
        if self.noisy:
            return self.rng.exponential(t)
        else:
            return t
        # results = []
        # for r in t:
        #     n = np.random.exponential(r)
        #     results += [math.log(n)]
        # return np.array(results)

    def name(self):
        if self.noisy:
            return "noisy"+self._problemID
        else:
            return self._problemID
        # return "IndTimeModel"