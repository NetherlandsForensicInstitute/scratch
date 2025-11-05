import arviz as az
import numpy as np
import pymc as pm
from scipy.stats import binom, betabinom


def analyse_bayesian(scores_eval, scores_km, scores_knm, stats_model):

    model_km = McmcModel(stats_model.km, stats_model.mcmc_settings).fit(scores_km)
    model_knm = McmcModel(stats_model.knm, stats_model.mcmc_settings).fit(scores_knm)

    logp_km = model_km.transform(scores_eval)
    logp_knm = model_knm.transform(scores_eval)

    return logp_km, logp_knm


class McmcModel:
    def __init__(self, stats_model, mcmc_settings):
        self.stats_model = stats_model
        self.mcmc_settings = mcmc_settings
        self.parameter_samples = None

    def fit(self, scores_obs: np.ndarray):
        parameters = list(self.stats_model['parameters'])

        with pm.Model():
            # Define the prior distributions of the model parameters
            priors = {}
            for parameter in parameters:
                parameter_input = self.stats_model.parameters[parameter]
                if parameter_input.prior == 'uniform':
                    prior = pm.Uniform(parameter, parameter_input.min, parameter_input.max)
                elif parameter_input.prior == 'beta':
                    prior = pm.Beta(parameter, parameter_input.a, parameter_input.b)
                else:
                    raise ValueError('Unrecognized prior')
                priors.update({parameter: prior})

            # Define the model: parameters and observed data
            if self.stats_model.distribution == 'betabinomial':
                pm.BetaBinomial('k', alpha=priors['alpha'], beta=priors['beta'], n=scores_obs[:,1], observed=scores_obs[:,0])
            elif self.stats_model.distribution == 'binomial':
                pm.Binomial('k', p=priors['p'], n=np.sum(scores_obs[:,1]), observed=np.sum(scores_obs[:,0]))
            else:
                raise ValueError('Unrecognized distribution')

            # Do simulations and sample from the posterior distributions
            trace = pm.sample(draws=self.mcmc_settings.draw_count, chains=self.mcmc_settings.chain_count,
                              tune=self.mcmc_settings.tune_count, cores=1, random_seed=None, progressbar=False)

            # Get the posterior samples of the model parameters, and print some (convergence) statistics
            self.parameter_samples = {}
            for parameter in parameters:
                samples = np.concatenate(np.array(trace.posterior[parameter]))
                self.parameter_samples.update({parameter: samples})
            summary = az.summary(trace, round_to=6)
            print(summary[['mean', 'sd', 'r_hat']])
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
        with pm.Model():
            if self.stats_model.distribution == 'betabinomial':
                logp = betabinom.logpmf(scores_2d[0], scores_2d[1], parameters_2d['alpha'], parameters_2d['beta'])
            elif self.stats_model.distribution == 'binomial':
                logp = binom.logpmf(scores_2d[0], scores_2d[1], parameters_2d['p'])

        # Return 10-bas log probabilities
        return logp / np.log(10)
