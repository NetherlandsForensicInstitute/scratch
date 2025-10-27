import arviz as az
import numpy as np
import pymc as pm
from scipy.stats import binom, betabinom

def analyse_bayesian(scores_eval, scores_km, scores_knm, stats_model):

    probs_km = run_mcmc_sims(scores_eval, scores_km, stats_model.km, stats_model.mcmc_settings)
    probs_knm = run_mcmc_sims(scores_eval, scores_knm, stats_model.knm, stats_model.mcmc_settings)
    llrs = np.log10(probs_km) - np.log10(probs_knm)

    return llrs, probs_km, probs_knm

def run_mcmc_sims(scores_eval, scores_obs, stats_model, mcmc_settings):

    sample_count = mcmc_settings.chain_count * mcmc_settings.draw_count
    k_obs_2d = np.tile(np.expand_dims(scores_eval[:,0], 1), (1, sample_count))
    n_obs_2d = np.tile(np.expand_dims(scores_eval[:,1], 1), (1, sample_count))

    with pm.Model():
        if stats_model.distribution == 'betabinomial':
            # Define the model: parameters and observed data
            prior_alpha = pm.Uniform('alpha', stats_model.parameters.alpha.min, stats_model.parameters.alpha.max)
            prior_beta = pm.Uniform('beta', stats_model.parameters.beta.min, stats_model.parameters.beta.max)
            pm.BetaBinomial('k', alpha=prior_alpha, beta=prior_beta, n=scores_obs[:,1], observed=scores_obs[:,0])
            # Do simulations and sample from the posterior distributions
            trace = pm.sample(draws=mcmc_settings.draw_count, chains=mcmc_settings.chain_count,
                              tune=mcmc_settings.tune_count, cores=1, random_seed=None, progressbar=False)
            # Get the posterior samples of the model parameters, and some (convergence) statistics
            post_alpha = np.concatenate(np.array(trace.posterior['alpha']))
            post_beta = np.concatenate(np.array(trace.posterior['beta']))
            # Calculate probabilities on specified score grid values
            post_alpha_2d = np.tile(np.expand_dims(post_alpha, 0), (len(scores_eval), 1))
            post_beta_2d = np.tile(np.expand_dims(post_beta, 0), (len(scores_eval), 1))
            probs = betabinom.pmf(k_obs_2d, n_obs_2d, post_alpha_2d, post_beta_2d)
        elif stats_model.distribution == 'binomial':
            # Define the model: parameters and observed data
            prior_p = pm.Beta('p', stats_model.parameters.p.a, stats_model.parameters.p.b)
            pm.Binomial('k', p=prior_p, n=np.sum(scores_obs[:,1]), observed=np.sum(scores_obs[:,0]))
            # Do simulations and sample from the posterior distributions
            trace = pm.sample(draws=mcmc_settings.draw_count, chains=mcmc_settings.chain_count,
                              tune=mcmc_settings.tune_count, cores=1, random_seed=None, progressbar=False)
            # Get the posterior samples of the model parameters, and some (convergence) statistics
            post_p = np.concatenate(np.array(trace.posterior['p']))
            # Calculate probabilities on specified score grid values
            post_p_2d = np.tile(np.expand_dims(post_p, 0), (len(scores_eval), 1))
            probs = binom.pmf(k_obs_2d, n_obs_2d, post_p_2d)

    summary = az.summary(trace, round_to=6)
    print(summary[['mean', 'sd', 'r_hat']])

    return probs
