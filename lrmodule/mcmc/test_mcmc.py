import confidence
import numpy as np
import pandas as pd

from lrmodule.mcmc.lr_analyses import analyse_bayesian, run_mcmc_sims

cfg = confidence.load_name("lrmodule/mcmc/config")

def test_mcmc_binomial():

    csv_prefix = 'lrmodule/mcmc/firing_pin_impression-' + cfg.dataset.score

    df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)
    scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])
    scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
    probs_eval = run_mcmc_sims(scores_eval, scores_knm, cfg.dataset.statistical_model.knm, cfg.dataset.statistical_model.mcmc_settings)
    probs = np.quantile(probs_eval, [cfg.interval.lower_bound, 0.5, cfg.interval.upper_bound], axis=1)

    df_probs = pd.read_csv(csv_prefix + '-knm_logp.csv', header=None)
    assert np.allclose(np.log10(probs), df_probs.to_numpy(), rtol=1E-1, atol=1E-1)

def test_llr_firing_pin():

    csv_prefix = 'lrmodule/mcmc/firing_pin_impression-' + cfg.dataset.score

    df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)
    scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])
    scores_km = np.int32(np.array(df_scores[df_scores[0]=='km'])[:,1:])
    scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
    llrs_eval, probs_km_eval, probs_knm_eval = analyse_bayesian(scores_eval, scores_km, scores_knm, cfg.dataset.statistical_model)
    llrs = np.quantile(llrs_eval, [cfg.interval.lower_bound, 0.5, cfg.interval.upper_bound], axis=1)

    df_llrs = pd.read_csv(csv_prefix + '-llrs-unbound.csv', header=None)
    assert np.allclose(llrs, df_llrs.to_numpy(), atol=1E-2)


def test_llr_breech_face():

    csv_prefix = 'lrmodule/mcmc/breech_face_impression-' + cfg.dataset.score

    df_scores = pd.read_csv(csv_prefix + '-all-input.csv', header=None)
    scores_eval = np.int32(np.array(df_scores[df_scores[0]=='eval'])[:,1:])
    scores_km = np.int32(np.array(df_scores[df_scores[0]=='km'])[:,1:])
    scores_knm = np.int32(np.array(df_scores[df_scores[0]=='knm'])[:,1:])
    llrs_eval, probs_km_eval, probs_knm_eval = analyse_bayesian(scores_eval, scores_km, scores_knm, cfg.dataset.statistical_model)
    llrs = np.quantile(llrs_eval, [cfg.interval.lower_bound, 0.5, cfg.interval.upper_bound], axis=1)

    df_llrs = pd.read_csv(csv_prefix + '-llrs-unbound.csv', header=None)
    assert np.allclose(llrs, df_llrs.to_numpy(), atol=1E-2)

print('finished')
