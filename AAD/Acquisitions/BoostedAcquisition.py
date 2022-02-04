# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.EI import AcquisitionEI
from GPyOpt.util.general import get_quantiles
import numpy as np
class BoostedAcquisition(AcquisitionBase):
    """
    """

    analytical_gradient_prediction = True

    def __init__(self, acquisition,parent_model=None):
        self.acquisition = acquisition
        if not parent_model is None:
            self.parent_acquisition = parent_model.acquisition_function #AcquisitionEI(parent_model,acquisition.space,acquisition.optimizer)
        else:
            self.parent_acquisition = None
        super(BoostedAcquisition, self).__init__(acquisition.model, acquisition.space, acquisition.optimizer)
        self.name = "Boosted_"+self.acquisition.name

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        """
        acq = self.acquisition._compute_acq(x)
        if(self.parent_acquisition is not None):
            acqu = self.parent_acquisition._compute_acq(x)
            # acqu = (-acqu - min(-acqu))/(max(-acqu - min(-acqu))+1e-3)
            acq = acq * acqu
            # embed()
        return acq

    def _compute_acq_withGradients(self, X):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        acq,dacq = self.acquisition._compute_acq_withGradients(X)
        if(self.parent_acquisition is not None):
            acqu,dacqu = self.parent_acquisition._compute_acq_withGradients(X)
            acq = acq * acqu
            dacq = dacq*acqu+acq*dacqu
            # acqu = -1*acqu; n = (max(-acqu - min(-acqu))+1e-3)
            # acqu = (-acqu - min(-acqu))/n

        return acq,dacq

