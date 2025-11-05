from typing import Self

import arviz as az
import confidence
import numpy as np
import pymc as pm

from lir.data.models import FeatureData, LLRData
from lir.transform import Transformer
from scipy.stats import binom, betabinom


class McmcLLRModel(Transformer):
    def __init__(self,
                 distribution_h1: str,
                 parameter_priors_h1: confidence.Configuration | None,
                 distribution_h2: str,
                 parameter_priors_h2: confidence.Configuration | None,
                 interval: tuple[float, float] = (0.05, 0.95),
                 ):
        self.model_h1 = McmcModel(distribution_h1, parameter_priors_h1)
        self.model_h2 = McmcModel(distribution_h2, parameter_priors_h2)
        self.interval = interval

    def fit(self, instances: FeatureData, **mcmc_kwargs) -> Self:
        self.model_h1.fit(instances.features[instances.labels==1], **mcmc_kwargs)
        self.model_h2.fit(instances.features[instances.labels==0], **mcmc_kwargs)

    def transform(self, instances: FeatureData) -> LLRData:
        logp_h1 = self.model_h1.transform(instances.features)
        logp_h2 = self.model_h2.transform(instances.features)
        llrs = logp_h1 - logp_h2
        quantiles = np.quantile(llrs, [.5] + list(self.interval), axis=1, method='midpoint')
        return LLRData(features=quantiles.transpose(1, 0))


class McmcModel:
    def __init__(self,
                 distribution: str,
                 parameters: confidence.Configuration | None,
                 ):
        self.distribution = distribution
        self.parameters = parameters
        self.chain_count = None
        self.tune_count = None
        self.draw_count = None
        self.random_seed = None
        self.parameter_samples = None
        self.r_hat = None

    def fit(self,
            scores_obs: np.ndarray,
            chain_count: int = 4,
            tune_count: int = 1000,
            draw_count: int = 1000,
            random_seed: int = None,
            ):
        self.chain_count = chain_count
        self.tune_count = tune_count
        self.draw_count = draw_count
        self.random_seed = random_seed

        # It looks like all pymc stuff needs to be in a single model block
        with pm.Model():
            # Define the prior distributions of the model parameters
            priors = {}
            for parameter in list(self.parameters.keys()):
                parameter_input = self.parameters[parameter]
                if parameter_input.prior == 'uniform':
                    prior = pm.Uniform(parameter, parameter_input.lower, parameter_input.upper)
                elif parameter_input.prior == 'beta':
                    prior = pm.Beta(parameter, alpha=parameter_input.alpha, beta=parameter_input.beta)
                else:
                    raise ValueError('Unrecognized prior')
                priors.update({parameter: prior})
            # Define the model: priors and the observed data
            if self.distribution == 'betabinomial':
                pm.BetaBinomial('k', alpha=priors['alpha'], beta=priors['beta'], n=scores_obs[:,1], observed=scores_obs[:,0])
            elif self.distribution == 'binomial':
                pm.Binomial('k', p=priors['p'], n=np.sum(scores_obs[:,1]), observed=np.sum(scores_obs[:,0]))
            else:
                raise ValueError('Unrecognized distribution')
            # Do simulations and sample from the posterior distributions
            trace = pm.sample(draws=self.draw_count, chains=self.chain_count,
                              tune=self.tune_count, cores=1, random_seed=self.random_seed, progressbar=False)
        # Get the posterior samples of the model parameters and convergence statistics
        self.parameter_samples = {}
        for parameter in list(self.parameters.keys()):
            samples = np.concatenate(np.array(trace.posterior[parameter]))
            self.parameter_samples.update({parameter: samples})
        summary = az.summary(trace, round_to=6)
        self.r_hat = summary['r_hat']
        return self

    def transform(self, scores_eval: np.ndarray) -> np.ndarray:
        # Prepare parameter and scores for 2d-evaluations
        sample_count = len(self.parameter_samples[list(self.parameter_samples.keys())[0]])
        scores_2d = {}
        for score_id in range(scores_eval.shape[1]):
            score_2d = np.tile(np.expand_dims(scores_eval[:,score_id], 1), (1, sample_count))
            scores_2d.update({score_id: score_2d})
        parameters_2d = {}
        for parameter in list(self.parameter_samples.keys()):
            parameter_2d = np.tile(np.expand_dims(self.parameter_samples[parameter], 0), (len(scores_eval), 1))
            parameters_2d.update({parameter: parameter_2d})
        # Calculate e-base log probabilities on specified scores values
        if self.distribution == 'betabinomial':
            logp = betabinom.logpmf(scores_2d[0], scores_2d[1], parameters_2d['alpha'], parameters_2d['beta'])
        elif self.distribution == 'binomial':
            logp = binom.logpmf(scores_2d[0], scores_2d[1], parameters_2d['p'])
        else:
            raise ValueError('Unrecognized distribution')
        # Return 10-base log probabilities
        return logp / np.log(10)
