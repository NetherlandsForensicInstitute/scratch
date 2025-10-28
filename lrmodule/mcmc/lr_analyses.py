import arviz as az
import numpy as np
import pymc as pm
from scipy.stats import binom, betabinom


def analyse_bayesian(scores_eval, scores_km, scores_knm, stats_model):

    parameters_km = fit_mcmc(stats_model.km, scores_km, stats_model.mcmc_settings)
    parameters_knm = fit_mcmc(stats_model.knm, scores_knm, stats_model.mcmc_settings)

    probs_km = apply_mcmc(stats_model.km, parameters_km, scores_eval)
    probs_knm = apply_mcmc(stats_model.knm, parameters_knm, scores_eval)

    llrs = np.log10(probs_km) - np.log10(probs_knm)

    return llrs, probs_km, probs_knm


def fit_mcmc(stats_model, scores_obs, mcmc_settings):

    parameters = list(stats_model['parameters'])

    with pm.Model():
        # Define the prior distributions of the model parameters
        priors = {}
        for parameter in parameters:
            parameter_input = stats_model.parameters[parameter]
            if parameter_input.prior == 'uniform':
                prior = pm.Uniform(parameter, parameter_input.min, parameter_input.max)
            elif parameter_input.prior == 'beta':
                prior = pm.Beta(parameter, parameter_input.a, parameter_input.b)
            else:
                raise ValueError('Unrecognized prior')
            priors.update({parameter: prior})

        # Define the model: parameters and observed data
        if stats_model.distribution == 'betabinomial':
            pm.BetaBinomial('k', alpha=priors['alpha'], beta=priors['beta'], n=scores_obs[:,1], observed=scores_obs[:,0])
        elif stats_model.distribution == 'binomial':
            # Define the model: parameters and observed data
            pm.Binomial('k', p=priors['p'], n=np.sum(scores_obs[:,1]), observed=np.sum(scores_obs[:,0]))
        else:
            raise ValueError('Unrecognized distribution')

        # Do simulations and sample from the posterior distributions
        trace = pm.sample(draws=mcmc_settings.draw_count, chains=mcmc_settings.chain_count,
                          tune=mcmc_settings.tune_count, cores=1, random_seed=None, progressbar=False)

        # Get the posterior samples of the model parameters, and print some (convergence) statistics
        parameter_samples = {}
        for parameter in parameters:
            samples = np.concatenate(np.array(trace.posterior[parameter]))
            parameter_samples.update({parameter: samples})
        summary = az.summary(trace, round_to=6)
        print(summary[['mean', 'sd', 'r_hat']])

    return parameter_samples


def apply_mcmc(stats_model, parameters, scores_eval):

    # Prepare parameter and scores for 2d-evaluations
    sample_count = len(parameters[list(parameters.keys())[0]])
    scores_2d = {}
    for score_id in range(scores_eval.shape[1]):
        score_2d = np.tile(np.expand_dims(scores_eval[:,score_id], 1), (1, sample_count))
        scores_2d.update({score_id: score_2d})
    parameters_2d = {}
    for parameter in list(parameters.keys()):
        parameter_2d = np.tile(np.expand_dims(parameters[parameter], 0), (len(scores_eval), 1))
        parameters_2d.update({parameter: parameter_2d})

    # Calculate e-base log probabilities on specified scores values
    with pm.Model():
        if stats_model.distribution == 'betabinomial':
            logp = betabinom.logpmf(scores_2d[0], scores_2d[1], parameters_2d['alpha'], parameters_2d['beta'])
        elif stats_model.distribution == 'binomial':
            # Calculate probabilities on specified score grid values
            logp = binom.logpmf(scores_2d[0], scores_2d[1], parameters_2d['p'])

    # Return 10-bas log probabilities
    return logp / np.log(10)
