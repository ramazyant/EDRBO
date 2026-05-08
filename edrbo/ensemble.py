from typing import List
from functools import reduce

from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR

from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood


class GPEnsemble(Model):
    _num_outputs: int = 1

    def __init__(self, *models: List[SingleTaskGP]):
        super().__init__()
        self.models = models

    @property
    def num_outputs(self):
        return self._num_outputs
    
    @property
    def num_inputs(self):
        return self.models[0].train_inputs[0].size(-1)

    def forward(self, *args, **kwargs):
        posteriors = [model(*args, **kwargs) for model in self.models]
        return reduce(lambda x, y: x + y, posteriors) / len(self.models)

    def posterior(self, *args, **kwargs):
        post_dist = self.forward(*args, **kwargs)
        return GPyTorchPosterior(post_dist)

    def fit_model(self):
        for model in self.models:
            model = model.double()
            mll = ExactMarginalLogLikelihood(model.likelihood, model).double()
            optim = AdamW(model.parameters(), lr=1e-2)
            sched = CyclicLR(optim, base_lr=1e-4, max_lr=1e-2, step_size_up=10, step_size_down=20)
            fit_gpytorch_mll_torch(mll, step_limit=110, optimizer=optim, scheduler=sched)
