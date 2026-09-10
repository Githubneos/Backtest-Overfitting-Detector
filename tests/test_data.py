import numpy as np
import pytest

from overfit_detector.data import make_synthetic_strategies


def test_shape_and_column_names():
    d = make_synthetic_strategies(n_strategies=6, n_periods=300, n_signal=2, seed=0)
    assert d.shape == (300, 6)
    assert list(d.columns[:2]) == ["signal_0", "signal_1"]
    assert d.columns[2] == "noise_0"


def test_seed_is_reproducible():
    a = make_synthetic_strategies(n_strategies=4, n_periods=200, seed=5)
    b = make_synthetic_strategies(n_strategies=4, n_periods=200, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_noise_columns_have_requested_mean_and_vol():
    d = make_synthetic_strategies(n_strategies=200, n_periods=5000, mu=0.001, sigma=0.02, seed=1)
    assert d.to_numpy().mean() == pytest.approx(0.001, abs=5e-5)
    assert d.to_numpy().std(ddof=1) == pytest.approx(0.02, rel=0.02)


def test_autocorr_is_produced_without_changing_unconditional_vol():
    phi = 0.5
    d = make_synthetic_strategies(
        n_strategies=100, n_periods=5000, mu=0.0, sigma=0.02, autocorr=phi, seed=2
    )
    x = d.to_numpy()
    lag1 = np.mean([np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(x.shape[1])])
    assert lag1 == pytest.approx(phi, abs=0.03)
    # Shocks are rescaled by sqrt(1 - phi^2), so sigma stays the target vol.
    assert x.std(ddof=1) == pytest.approx(0.02, rel=0.03)


def test_signal_columns_actually_have_an_edge():
    d = make_synthetic_strategies(
        n_strategies=10, n_periods=4000, mu=0.0, signal_mu=0.002, n_signal=3, seed=3
    )
    assert (d.iloc[:, :3].mean() > 0.0015).all()
    assert d.iloc[:, 3:].mean().abs().max() < 0.0015


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(autocorr=1.0),
        dict(autocorr=-1.5),
        dict(sigma=0.0),
        dict(n_signal=99),
        dict(n_periods=1),
    ],
)
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        make_synthetic_strategies(n_strategies=5, n_periods=kwargs.pop("n_periods", 100), **kwargs)


def test_yfinance_loader_is_importable_and_lazy():
    """The optional dependency must not be needed to import the package."""
    from overfit_detector import load_yfinance_strategies

    assert callable(load_yfinance_strategies)
