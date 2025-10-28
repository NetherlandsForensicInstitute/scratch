import confidence
import numpy as np
import pandas as pd

from lrmodule.mcmc.lr_analyses import analyse_bayesian, fit_mcmc, apply_mcmc

cfg = confidence.load_name("lrmodule/mcmc/config")


# def test_fit_mcmc_parameter_stats():
#
#     csv_prefix = 'lrmodule/mcmc/firing_pin_impression-' + cfg.dataset.score
#     df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)
#
#     # beta-binomial model
#     scores_km = np.int32(np.array(df_scores[df_scores[0]=='km'])[:,1:])
#     parameters_km = fit_mcmc(cfg.dataset.statistical_model.km, scores_km, cfg.dataset.statistical_model.mcmc_settings)
#     stats_calc = np.array([[np.mean(parameters_km['alpha']), np.std(parameters_km['alpha'])],
#                            [np.mean(parameters_km['beta']), np.std(parameters_km['beta'])]])
#     stats_ref = np.loadtxt(csv_prefix + '-km_parameter_stats.csv', delimiter=',')
#     assert np.allclose(stats_calc, stats_ref, rtol=1E-2, atol=1E-2)
#
#     # binomial model
#     scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
#     parameters_knm = fit_mcmc(cfg.dataset.statistical_model.knm, scores_knm, cfg.dataset.statistical_model.mcmc_settings)
#     stats_calc = np.array([np.mean(parameters_knm['p']), np.std(parameters_knm['p'])])
#     stats_ref = np.loadtxt(csv_prefix + '-knm_parameter_stats.csv', delimiter=',')
#     assert np.allclose(stats_calc, stats_ref, rtol=1E-2, atol=1E-2)

def test_apply_mcmc_binomial():

    csv_prefix = 'lrmodule/mcmc/firing_pin_impression-' + cfg.dataset.score#
    df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)
    scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])

    # betabinomial model
    parameters_km = {'alpha': np.loadtxt(csv_prefix + '-km_parameters.csv', delimiter=',', max_rows=1),
                     'beta': np.loadtxt(csv_prefix + '-km_parameters.csv', delimiter=',', skiprows=1)}
    logp_eval = apply_mcmc(cfg.dataset.statistical_model.km, parameters_km, scores_eval)
    logp_calc = np.quantile(logp_eval, [cfg.interval.lower_bound, 0.5, cfg.interval.upper_bound], axis=1)
    logp_ref = np.loadtxt(csv_prefix + '-km_logp.csv', delimiter=',')
    assert np.allclose(logp_calc, logp_ref, rtol=1E-2, atol=1E-2)

    # binomial model
    parameters_knm = {'p': np.loadtxt(csv_prefix + '-knm_parameters.csv', delimiter=',')}
    logp_eval = apply_mcmc(cfg.dataset.statistical_model.knm, parameters_knm, scores_eval)
    logp_calc = np.quantile(logp_eval, [cfg.interval.lower_bound, 0.5, cfg.interval.upper_bound], axis=1)
    logp_ref = np.loadtxt(csv_prefix + '-knm_logp.csv', delimiter=',')
    assert np.allclose(logp_calc, logp_ref, rtol=1E-2, atol=1E-2)

# def test_llr_firing_pin():
#
#     csv_prefix = 'lrmodule/mcmc/firing_pin_impression-' + cfg.dataset.score
#
#     df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)
#     scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])
#     scores_km = np.int32(np.array(df_scores[df_scores[0]=='km'])[:,1:])
#     scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
#     llrs_eval, probs_km_eval, probs_knm_eval = analyse_bayesian(scores_eval, scores_km, scores_knm, cfg.dataset.statistical_model)
#     llrs = np.quantile(llrs_eval, [cfg.interval.lower_bound, 0.5, cfg.interval.upper_bound], axis=1)
#
#     df_llrs = pd.read_csv(csv_prefix + '-llr_unbound.csv', header=None)
#     assert np.allclose(llrs, df_llrs.to_numpy(), rtol=1E-1, atol=1E-1)
#
#
# def test_llr_breech_face():
#
#     csv_prefix = 'lrmodule/mcmc/breech_face_impression-' + cfg.dataset.score
#
#     df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)
#     scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])
#     scores_km = np.int32(np.array(df_scores[df_scores[0]=='km'])[:,1:])
#     scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
#     llrs_eval, probs_km_eval, probs_knm_eval = analyse_bayesian(scores_eval, scores_km, scores_knm, cfg.dataset.statistical_model)
#     llrs = np.quantile(llrs_eval, [cfg.interval.lower_bound, 0.5, cfg.interval.upper_bound], axis=1)
#
#     df_llrs = pd.read_csv(csv_prefix + '-llr_unbound.csv', header=None)
#     assert np.allclose(llrs, df_llrs.to_numpy(), rtol=1E-1, atol=1E-1)

print('finished')
