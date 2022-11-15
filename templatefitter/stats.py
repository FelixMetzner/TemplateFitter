"""
Provides statistical methods for goodness of fit test
"""

import numpy as np

from scipy.stats import chi2
from typing import Optional, Union, Tuple, List

__all__ = [
    "pearson_chi2_test",
    "cowan_binned_likelihood_gof",
    "toy_chi2_test",
    "ToyInfoOutputType",
]

ToyInfoOutputType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def pearson_chi2_test(
    data: np.ndarray,
    expectation: np.ndarray,
    dof: int,
    error: Optional[np.ndarray] = None,
) -> Tuple[float, int, float]:
    """
    Performs a Pearson chi2-test.
    This test reflects the level of agreement between observed and expected histograms.
    The test statistic is

        chi2 = sum limits(i=1; n_bins) (n_i - nu_i)^2 / nu_i,

    where n_i is the number of observations in bin i and nu_i is the expected number of events in bin i.

    In the large sample limits, this test statistic follows a chi2-distribution
    with n_bins - m degrees of freedom,
    where m is the number of unconstrained fit parameters.

    Parameters
    ----------
    data : np.ndarray
        Data bin counts. Shape is (num_bins,)
    expectation : np.ndarray
        Expected bin counts. Shape is (num_bins,)
    dof : int
        Degrees of freedom. This is the number of bins minus the number of free fit parameters.

    Returns
    -------
    float
        chi2
    float
        dof
    float
        p-value
    """

    if error is not None:
        chi_sq = np.sum((data - expectation) ** 2 / error)
    else:
        chi_sq = np.sum((data - expectation) ** 2 / expectation)

    assert isinstance(chi_sq, float), type(chi_sq)
    p_val = chi2.sf(chi_sq, df=dof)
    return chi_sq, dof, p_val


def cowan_binned_likelihood_gof(
    data: np.ndarray,
    expectation: np.ndarray,
    dof: int,
) -> Tuple[float, int, float]:
    """
    Performs a GOF-test using a test statistic based on a
    binned likelihood function.
    The test statistic is the ratio lambda(nu) = L(nu=hat(nu)) / L(theta=n),
    where nu are the expected values in each bin.
    In the numerator (denominator), the likelihood is evaluated with the estimated values for nu (the measured values).

    In the large sample limit, the test statistic

        chi2 = -2 log lambda = 2 sum limits(i=1; n_bins) n_i log(n_i / hat(nu_i)) - hat(nu_i) - n_i,

    follows a chi2-distribution with n_bins - m
    degrees of freedom, where m is the number of unconstrained
    fit parameters.

    Parameters
    ----------
    data : np.ndarray
        Data bin counts. Shape is (num_bins,)
    expectation : np.ndarray
        Expected bin counts. Shape is (num_bins,)
    dof : int
        Degrees of freedom. This is the number of bins minus the
        number of free fit parameters.

    Returns
    -------
    float
        chi2
    float
        dof
    float
        p-value
    """

    chi_sq = 2 * np.sum(data * np.log(data / expectation) + expectation - data)

    assert isinstance(chi_sq, float), type(chi_sq)
    p_val = chi2.sf(chi_sq, df=dof)
    return chi_sq, dof, p_val


def calc_chi_squared(
    obs: np.ndarray,
    exp: np.ndarray,
    exp_unc: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Calculates the chi squared difference between an expected and an observed histogrammed distribution.
    If obs is 2-dimensional (contains multiple histograms), an array of chi squared values will be returned.

    Parameters
    ----------
    obs: np.ndarray
        Array containing histogrammed observed data for which the distribution
        shall be compared to the expected distribution. Shape is (len(exp),) or
        (len(exp), >=1), where len(exp) is the number of bins of the histogrammed
        expected distribution.
    exp: np.ndarray
        Array containing the histogrammed expected distribution. Shape is (num_bins, ).
    exp_unc: np.ndarray
        Array containing the uncertainty on the bins of the histogrammed expected
        distribution. Shape is (num_bins, )
    Returns
    -------
    float or np.ndarray
        Resulting chi squared value or array of chi squared values.
    """
    if len(obs.shape) > 1:
        return np.sum(np.nan_to_num((obs - exp) ** 2 / exp_unc), axis=1)
    else:
        return np.sum(np.nan_to_num((obs - exp) ** 2 / exp_unc))


def calc_chi_squared_with_cov(
    obs: np.ndarray,
    exp: np.ndarray,
    exp_cov: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Calculates the chi squared difference between an expected and an observed histogrammed distribution.
    In this calculation the covariance matrix of the expectation is considered.
    If obs is 2-dimensional (contains multiple histograms), an array of chi squared values will be returned.

    Parameters
    ----------
    obs: np.ndarray
        Array containing histogrammed observed data for which the distribution
        shall be compared to the expected distribution. Shape is (len(exp),) or
        (len(exp), >=1), where len(exp) is the number of bins of the histogrammed
        expected distribution.
    exp: np.ndarray
        Array containing the histogrammed expected distribution. Shape is (num_bins, ).
    exp_cov: np.ndarray
        2D-Array containing the covariance matrix of the histogrammed expected distribution. Shape (num_bins, num_bins)
    Returns
    -------
    float or np.ndarray
        Resulting chi squared value or array of chi squared values.
    """
    diff = obs - exp  # type: np.ndarray
    inverse_cov = np.linalg.inv(exp_cov)  # type: np.ndarray
    if len(obs.shape) > 1:
        return np.sum(np.tensordot(diff, inverse_cov, axes=1) * diff, axis=1)
    else:
        return diff.T @ inverse_cov @ diff


def mc_chi_squared_from_toys(
    obs: np.ndarray,
    exp: np.ndarray,
    exp_unc: np.ndarray,
    mc_cov: Optional[np.ndarray] = None,
    use_text_book_approach: bool = False,
    toys_size: int = 1000000,
    seed: int = 13377331,
) -> Tuple[float, np.ndarray]:
    """
    Evaluates chi squared difference of expected and observed histogrammed
    distributions and obtains the chi squared distribution for exp via toy
    samples. The number of toy samples for this evaluation can be set via
    the parameter toys_size.

    Parameters
    ----------
    obs: np.ndarray
        Histogrammed observed distribution.
    exp: np.ndarray
        Histogrammed expected distribution.
    exp_unc: np.ndarray
        Uncertainty on bins of histogrammed expected distribution.
    toys_size: int
        Size of toy sample to be produced and used to obtain the chi squared distribution to exp.
    seed: int
        Seed for random generator to be used to create reproducible toy samples. Default is 13377331.

    Returns
    -------
    float
        Chi squared value of obs and exp comparison.
    np.ndarray
        Sampled chi squared values obtained from exp via toys.
    """
    exp_ge_zero = exp > 0  # type: np.ndarray
    _obs = obs[exp_ge_zero]  # type: np.ndarray
    _exp = exp[exp_ge_zero]  # type: np.ndarray
    _exp_unc = exp_unc[exp_ge_zero]  # type: np.ndarray

    _mc_cov = np.array([])  # type: np.ndarray
    has_valid_cov = not (mc_cov is None or not np.any(mc_cov))  # type: bool

    if has_valid_cov:
        _mc_cov = mc_cov
        if not _exp.shape[0] == _mc_cov.shape[0] == _mc_cov.shape[1]:
            assert len(exp_ge_zero.shape) == 1, exp_ge_zero.shape
            exp_ge_zero_indices = [i for i, is_not in enumerate(exp_ge_zero) if is_not]  # type: List[int]
            _mc_cov = _mc_cov[np.ix_(exp_ge_zero_indices, exp_ge_zero_indices)]

    if use_text_book_approach:
        assert has_valid_cov, (type(mc_cov), mc_cov)
        _obs_chi_squared = calc_chi_squared_with_cov(obs=_obs, exp=_exp, exp_cov=_mc_cov)
        assert isinstance(_obs_chi_squared, float), (type(_obs_chi_squared), _obs_chi_squared)
        obs_chi_squared = _obs_chi_squared  # type: float
    else:
        _obs_chi_squared = calc_chi_squared(obs=_obs, exp=_exp, exp_unc=_exp_unc)
        assert isinstance(_obs_chi_squared, float), (type(_obs_chi_squared), _obs_chi_squared)
        obs_chi_squared = _obs_chi_squared

    random_state = np.random.RandomState(seed=seed)  # type: np.random.RandomState

    if has_valid_cov and not use_text_book_approach:
        toys = random_state.multivariate_normal(mean=_exp, cov=_mc_cov, size=toys_size)  # type: np.ndarray
        toys[toys < 0.0] = 0.0
    else:
        # Use poisson approach for textbook approach or if no valid covariance matrix is provided.
        toys = random_state.poisson(_exp, size=(toys_size, len(_exp)))

    if use_text_book_approach:
        toy_chi_squared = calc_chi_squared_with_cov(obs=toys, exp=_exp, exp_cov=_mc_cov)  # type: np.ndarray
    else:
        toy_chi_squared = calc_chi_squared(obs=toys, exp=_exp, exp_unc=_exp_unc)

    assert np.min(toy_chi_squared) < np.max(toy_chi_squared), (np.min(toy_chi_squared), np.max(toy_chi_squared))
    return obs_chi_squared, toy_chi_squared


def _toy_chi2_test(
    data: np.ndarray,
    expectation: np.ndarray,
    error: np.ndarray,
    mc_cov: Optional[np.ndarray] = None,
    use_text_book_approach: bool = False,
    toys_size: int = 1000000,
) -> Tuple[float, float, ToyInfoOutputType]:
    obs_chi2, toys = mc_chi_squared_from_toys(
        obs=data,
        exp=expectation,
        exp_unc=error,
        mc_cov=mc_cov,
        use_text_book_approach=use_text_book_approach,
        toys_size=toys_size,
    )

    bc, be = np.histogram(toys, bins=100, density=True)
    bm = (be[1:] + be[:-1]) / 2
    bw = be[1:] - be[:-1]

    p_val = np.sum(bc[bm > obs_chi2] * bw[bm > obs_chi2])
    assert isinstance(p_val, float)

    return obs_chi2, p_val, (bc, be, toys)


def toy_chi2_test(
    data: np.ndarray,
    expectation: np.ndarray,
    error: np.ndarray,
    mc_cov: Optional[np.ndarray] = None,
    use_text_book_approach: bool = False,
    toys_size: int = 1000000,
    max_attempts: int = 3,
) -> Tuple[float, float, ToyInfoOutputType]:
    """
    Performs a GoF-test using a test statistic based on toy MC sampled
    from the expected distribution.

    Parameters
    ----------
    data : np.ndarray
        Data bin counts. Shape is (num_bins,)
    expectation : np.ndarray
        Expected bin counts. Shape is (num_bins,)
    error : np.ndarray
        Uncertainty on the expected distribution. Shape is (num_bins,)
    mc_cov: 2D np.ndarray
    toys_size : int
        Number of toy samples to be drawn from expectation to model the chi2 of the expectation. Default is 1000000.
    max_attempts: Maximal number of tries, each decreasing the toy-size by a factor of 10.

    Returns
    -------
    float
        chi2
    float
        p-value.
    tuple(bin_counts, bin_edges, chi2_toys)
        Information needed to plot the chi2 distribution obtained from the toys.
    """

    try_count = 0
    while try_count < max_attempts:
        try:
            return _toy_chi2_test(
                data=data,
                expectation=expectation,
                error=error,
                mc_cov=mc_cov,
                use_text_book_approach=use_text_book_approach,
                toys_size=int(toys_size * 0.1 ** try_count),
            )
        except IndexError as ie:
            if try_count == max_attempts - 1:
                raise ie
        try_count += 1

    return 0.0, 0.0, (np.array([]), np.array([]), np.array([]))
