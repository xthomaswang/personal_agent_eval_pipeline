import numpy as np

from wsp import metrics


def test_topk_orders_and_breaks_ties_by_index():
    scores = np.array([[0.1, 0.9, 0.9, 0.5], [0.3, 0.2, 0.1, 0.0]])
    top = metrics.topk(scores, 3)
    assert top[0].tolist() == [1, 2, 3]
    assert top[1].tolist() == [0, 1, 2]
    assert metrics.topk(scores, 10).shape == (2, 4)


def test_recall_mrr_ndcg_on_toy_ranking():
    ranking = np.array([5, 2, 7, 1, 9])
    rel = {2: 1.0, 9: 1.0, 4: 1.0}
    assert metrics.recall_at_k(ranking, rel, 2) == 1 / 3
    assert metrics.recall_at_k(ranking, rel, 5) == 2 / 3
    assert metrics.recall_any_at_k(ranking, rel, 1) == 0.0
    assert metrics.recall_any_at_k(ranking, rel, 2) == 1.0
    assert metrics.mrr_at_k(ranking, rel, 10) == 1 / 2
    assert metrics.mrr_at_k(ranking, rel, 1) == 0.0
    dcg = 1 / np.log2(3) + 1 / np.log2(6)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert np.isclose(metrics.ndcg_at_k(ranking, rel, 5), dcg / idcg)


def test_ndcg_graded_gain_and_perfect_ranking():
    rel = {0: 2.0, 1: 1.0}
    assert np.isclose(metrics.ndcg_at_k(np.array([0, 1, 2]), rel, 10), 1.0)
    worse = metrics.ndcg_at_k(np.array([1, 0, 2]), rel, 10)
    assert 0 < worse < 1


def test_session_recall():
    sessions = ["a", "a", "b", "c"]
    frac, any_ = metrics.session_recall_at_k(np.array([0, 1, 2]), sessions, {"a", "c"}, 3)
    assert (frac, any_) == (0.5, 1.0)
    assert metrics.session_recall_at_k(np.array([2]), sessions, {"a"}, 5) == (0.0, 0.0)


def test_query_metrics_have_the_documented_names():
    r = metrics.query_metrics("lme", np.array([0]), {0: 1.0}, ["s"], {"s"})
    assert set(r) == set(metrics.METRICS_BY_KIND["lme"])
    assert set(metrics.query_metrics("mteb", np.array([0]), {0: 1.0})) == set(metrics.METRICS_BY_KIND["mteb"])
    assert set(metrics.query_metrics("locomo", np.array([0]), {0: 1.0})) == set(metrics.METRICS_BY_KIND["locomo"])


def test_bootstrap_ci_contains_the_mean_and_excludes_zero_for_a_clear_gain():
    diff = np.full(50, 0.1) + np.random.default_rng(0).normal(0, 0.01, 50)
    lo, hi = metrics.bootstrap_ci(diff, n_boot=2000)
    assert lo < diff.mean() < hi and lo > 0
