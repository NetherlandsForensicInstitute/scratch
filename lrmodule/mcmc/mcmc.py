from typing import Self

import arviz as az
import numpy as np
import pymc as pm

from lir.data.models import FeatureData, LLRData
from lir.transform import Transformer
from scipy.stats import binom, betabinom


class McmcLLRModel(Transformer):
    """
    Use Markov Chain Monte Carlo simulations to fit a statistical distribution for each of the two hypotheses. Using
    samples from the posterior distributions of the model parameters, a posterior distribution of the LR is obtained.
    The median of this distribution is used as best estimate for the LR; a credible interval is also determined.
    """

    def __init__(self,
                 distribution_h1: str,
                 parameters_h1: dict[str, dict] | None,
                 distribution_h2: str,
                 parameters_h2: dict[str, dict] | None,
                 interval: tuple[float, float] = (0.05, 0.95),
                 ):
        """
        :param distribution_h1: statistical distribution used to model H1
        :param parameters_h1: definition of the parameters of distribution_h1, and their prior distributions
        :param distribution_h2: statistical distribution used to model H2
        :param parameters_h2: definition of the parameters of distribution_h2, and their prior distributions
        :param interval: lower and upper bounds of the credible interval
        """
        self.model_h1 = McmcModel(distribution_h1, parameters_h1)
        self.model_h2 = McmcModel(distribution_h2, parameters_h2)
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
                 parameters: dict[str, dict] | None,
                 ):
        """
        :param distribution: statistical distribution used to model
        :param parameters: definition of the parameters of the distribution, and their prior distributions; it should
        be a dictionary where the keys are the parameter names used for the statistical distribution in pymc,
        and the values are dictionaries with a key 'prior', defining the prior distribution used for that parameter,
        and with additional keys corresponding to the parameter names used for that prior distribution.
        For example, for a binomial distribution: parameters = {'p': {'prior': 'beta', 'alpha': 0.5, 'beta': 0.5}}.
        """
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
        """
        :param scores_obs: observations, based on those the prior distributions of the parameters are updated
        :param chain_count: number of parallel mcmc chains
        :param tune_count: number of tune/warm-up/burn-in samples per chain
        :param draw_count: number of samples to draw from each chain
        :param random_seed: random seed
        """
        self.chain_count = chain_count
        self.tune_count = tune_count
        self.draw_count = draw_count
        self.random_seed = random_seed

        # It looks like all pymc stuff needs to be in a single model block
        with pm.Model():
            # Define the prior distributions of the model parameters based on their definitions
            priors = {}
            for parameter in list(self.parameters.keys()):
                parameter_input = self.parameters[parameter]
                if parameter_input['prior'] == 'uniform':
                    prior = pm.Uniform(parameter, parameter_input['lower'], parameter_input['upper'])
                elif parameter_input['prior'] == 'beta':
                    prior = pm.Beta(parameter, alpha=parameter_input['alpha'], beta=parameter_input['beta'])
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
