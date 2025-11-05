import confidence
import numpy as np
import pandas as pd

from lrmodule.mcmc.lr_analyses import analyse_bayesian, McmcModel

# Tests against references from the Matlab implementation.

cfg = confidence.load_name("lrmodule/mcmc/config")
quantages = [cfg.interval.lower_bound, 0.5, cfg.interval.upper_bound]


def test_fit_mcmc_parameter_stats():

    # Conversion from observed score data to (statistics of) model parameter samples.
    # We expect some differences here due to mcmc-differences and randomness.

    csv_prefix = 'lrmodule/mcmc/firing_pin_impression-' + cfg.dataset.score
    df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)

    # beta-binomial model
    scores_km = np.int32(np.array(df_scores[df_scores[0]=='km'])[:,1:])
    model_km = McmcModel(cfg.dataset.statistical_model.km, cfg.dataset.statistical_model.mcmc_settings).fit(scores_km)
    stats_calc = np.array([[np.mean(model_km.parameter_samples['alpha']), np.std(model_km.parameter_samples['alpha'])],
                           [np.mean(model_km.parameter_samples['beta']), np.std(model_km.parameter_samples['beta'])]])
    stats_ref = np.loadtxt(csv_prefix + '-km_parameter_stats.csv', delimiter=',')
    assert np.allclose(stats_calc, stats_ref, rtol=2E-2, atol=2E-2)

    # binomial model
    scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
    model_knm = McmcModel(cfg.dataset.statistical_model.knm, cfg.dataset.statistical_model.mcmc_settings).fit(scores_knm)
    stats_calc = np.array([np.mean(model_knm.parameter_samples['p']), np.std(model_knm.parameter_samples['p'])])
    stats_ref = np.loadtxt(csv_prefix + '-knm_parameter_stats.csv', delimiter=',')
    assert np.allclose(stats_calc, stats_ref, rtol=1E-4, atol=1E-4)

def test_transform_mcmc_binomial():

    # Conversion from model parameter samples to (percentiles of) log10-probabilities.

    csv_prefix = 'lrmodule/mcmc/firing_pin_impression-' + cfg.dataset.score#
    df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)
    scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])

    # betabinomial model
    alpha, beta = np.loadtxt(csv_prefix + '-km_parameter_samples.csv', delimiter=',', unpack=True)
    model_km = McmcModel(cfg.dataset.statistical_model.km, None)
    model_km.parameter_samples = {'alpha': alpha, 'beta': beta}
    logp_eval = model_km.transform(scores_eval)
    logp_calc = np.quantile(logp_eval, quantages, axis=1, method='midpoint')
    logp_ref = np.loadtxt(csv_prefix + '-km_logp.csv', delimiter=',')
    assert np.allclose(logp_calc, logp_ref)

    # binomial model
    model_knm = McmcModel(cfg.dataset.statistical_model.knm, None)
    model_knm.parameter_samples = {'p': np.loadtxt(csv_prefix + '-knm_parameter_samples.csv', delimiter=',')}
    logp_eval = model_knm.transform(scores_eval)
    logp_calc = np.quantile(logp_eval, quantages, axis=1, method='midpoint')
    logp_ref = np.loadtxt(csv_prefix + '-knm_logp.csv', delimiter=',')
    assert np.allclose(logp_calc, logp_ref)

def test_llr_firing_pin():

    # From pre-calculated scores to (percentiles of) log-10 lrs.
    # We expect some differences here due to mcmc-differences and randomness.

    csv_prefix = 'lrmodule/mcmc/firing_pin_impression-' + cfg.dataset.score
    df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)

    scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])
    scores_km = np.int32(np.array(df_scores[df_scores[0]=='km'])[:,1:])
    scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
    logp_km, logp_knm = analyse_bayesian(scores_eval, scores_km, scores_knm, cfg.dataset.statistical_model)
    llrs_calc = np.quantile(logp_km - logp_knm, quantages, axis=1, method='midpoint')
    llrs_ref = np.loadtxt(csv_prefix + '-llr_unbound.csv', delimiter=',')
    assert np.allclose(llrs_calc, llrs_ref, rtol=5E-2, atol=5E-2)


def test_llr_breech_face():

    # From pre-calculated scores to (percentiles of) log-10 lrs.
    # We expect some differences here due to mcmc-differences and randomness.

    csv_prefix = 'lrmodule/mcmc/breech_face_impression-' + cfg.dataset.score
    df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)

    scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])
    scores_km = np.int32(np.array(df_scores[df_scores[0]=='km'])[:,1:])
    scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
    logp_km, logp_knm = analyse_bayesian(scores_eval, scores_km, scores_knm, cfg.dataset.statistical_model)
    llrs_calc = np.quantile(logp_km - logp_knm, quantages, axis=1, method='midpoint')
    llrs_ref = np.loadtxt(csv_prefix + '-llr_unbound.csv', delimiter=',')
    assert np.allclose(llrs_calc, llrs_ref, rtol=5E-2, atol=5E-2)
