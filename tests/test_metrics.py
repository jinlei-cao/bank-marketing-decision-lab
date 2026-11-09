import numpy as np
import pytest

from banklab.metrics import classification_metrics, illustrative_scenario, ranking_at_capacity


def test_top_k_ordering_and_lift():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([.9, .2, .8, .1, .3])
    m = ranking_at_capacity(y, p, .4)
    assert m["contacts"] == 2
    assert m["observed_subscriptions"] == 2
    assert m["precision_at_k"] == 1.0
    assert m["lift_at_k"] == pytest.approx(2.5)


def test_scenario_is_hypothetical_not_uplift():
    top = ranking_at_capacity([1, 0, 1, 0], [.9, .2, .8, .1], .5)
    scen = illustrative_scenario(top, 50, 5)
    assert scen["assumed_gross_value"] == 100
    assert scen["assumed_contact_cost"] == 10
    assert scen["illustrative_net_proxy"] == 90


def test_metrics_are_valid():
    m = classification_metrics([0, 1, 0, 1], [.1, .9, .2, .8], .5)
    assert m["roc_auc"] == 1
    assert m["average_precision"] == 1
