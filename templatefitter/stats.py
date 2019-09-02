import numpy as np
from scipy.stats import chi2
from scipy.integrate import quad

__all__ = [
    "pearson_chi2_test",
    "cowan_binned_likelihood_gof",
    "toy_chi2_test"
]


# -- goodness of fit statistics --


def pearson_chi2_test(data, expectation, dof, error=None):
    """
    Performs a Pearson :math:`\chi^2`-test.
    This test reflects the level of agreement between observed
    and expected histograms.
    The test statistic is

    .. math::

        \chi^2=\sum\limits_{i=1}^{n_\mathrm{bins}} \\frac{(n_i - \\nu_i)^2}{\\nu_i},

    where :math:`n_i` is the number of observations in bin
    :math:`i` and :math:`\\nu_i` is the expected number of
    events in bin :math:`i`.

    In the large sample limits, this test statistic follows a
    :math:`\chi^2`-distribution with :math:`n_\mathrm{bins} - m`
    degrees of freedom, where :math:`m` is the number of unconstrained
    fit parameters.

    Parameters
    ----------
    data : np.ndarray
        Data bin counts. Shape is (`num_bins`,)
    expectation : np.ndarray
        Expected bin counts. Shape is (`num_bins`,)
    dof : int
        Degrees of freedom. This is the number of bins minus the
        number of free fit parameters.

    Returns
    -------
    float
        :math:`\chi^2`
    float
        :math:`\mathrm{dof}`
    float
        p-value.
    """

    if error is not None:
        chi_sq = np.sum((data - expectation) ** 2 / error)
    else:
        chi_sq = np.sum((data - expectation) ** 2 / expectation)

    pval = quad(chi2.pdf, chi_sq, np.inf, args=(dof,))[0]
    return chi_sq, dof, pval


def cowan_binned_likelihood_gof(data, expectation, dof):
    """
    Performs a GOF-test using a test statistic based on a
    binned likelihood function.
    The test statistic is the ratio :math:`\lambda(\\nu) = L(\\nu=\hat{\\nu})/L(\\theta=n)`,
    where :math:`\\nu` are the expected values in each bin. In the
    numerator (denominator), the likelihood is evaluated with the estimated
    values for :math:`\\nu` (the measured values).

    In the large sample limit, the test statistic

    .. math::

        \chi^2 = -2\log \lambda = 2\sum\limits_{i=1}^{n_\mathrm{bins}} n_i\log(\\frac{n_i}{\hat{\\nu_i}}) - \hat{\\nu_i} - n_i,

    follows a :math:`\chi^2`-distribution with :math:`n_\mathrm{bins} - m`
    degrees of freedom, where :math:`m` is the number of unconstrained
    fit parameters.

    Parameters
    ----------
    data : np.ndarray
        Data bin counts. Shape is (`num_bins`,)
    expectation : np.ndarray
        Expected bin counts. Shape is (`num_bins`,)
    dof : int
        Degrees of freedom. This is the number of bins minus the
        number of free fit parameters.

    Returns
    -------
    float
        :math:`\chi^2`
    float
        :math:`\mathrm{dof}`
    float
        p-value.
    """

    chi_sq = 2 * np.sum(data * np.log(data / expectation) + expectation - data)
    pval = quad(chi2.pdf, chi_sq, np.inf, args=(dof,))[0]
    return chi_sq, dof, pval


def calc_chi_squared(obs, exp, exp_unc):
    """
    Calculates the chi squared difference between an expected and an observed
    histogrammed distribution.
    If obs is 2-dimensional (contains multiple histogramms), an array of
    chi squared values will be returned.

    Parameters
    ----------
    obs: np.ndarray
        Array containing histogrammed observed data for which the distribution
        shall be compared to the expected distribution. Shape is (len(exp),) or
        (len(exp), >=1), where len(exp) is the number of bins of the histogrammed
        expected distribution.
    exp: np.ndarray
        Array containing the histogrammed expected distribution. Shape is (`num_bins`, ).
    exp_unc: np.ndarray
        Array containing the uncertainty on the bins of the histogrammed expected
        distribution. Shape is (`num_bins`, )
    Returns
    -------
    float or np.ndarray
        Resulting chi squared value or array of chi squared values.
    """
    if len(obs.shape) > 1:
        return np.sum(np.nan_to_num((obs - exp) ** 2 / exp_unc), axis=1)
    else:
        return np.sum(np.nan_to_num((obs - exp) ** 2 / exp_unc))


def mc_chi_squared_from_toys(obs, exp, exp_unc, mc_cov=None, toys_size=1000000, seed=13377331):
    """
    Evaluates chi squared difference of expected and observed histogrammed
    distributions and obtains the chi squared distribution for exp via toy
    samples. The number of toy samples for this evaluation can be set via
    the parameter `toys_size`.

    Parameters
    ----------
    obs: np.ndarray
        Histogrammed observed distribution.
    exp: np.ndarray
        Histogrammed expected distribution.
    exp_unc: np.ndarray
        Uncertainty on bins of histogrammed expected distribution.
    toys_size: int
        Size of toy sample to be produced and used to obtain the
        chi squared distribution to exp.
    seed: int
        Seed for random generator to be used to create reproducible
        toy samples. Default is 13377331.

    Returns
    -------
    float
        Chi squared value of obs and exp comparison.
    np.ndarray
        Sampled chi squared values obtained from exp via toys.
    """
    exp_ge_zero = exp > 0
    obs = obs[exp_ge_zero]
    exp = exp[exp_ge_zero]
    exp_unc = exp_unc[exp_ge_zero]

    obs_chi_squared = calc_chi_squared(obs, exp, exp_unc)

    np.random.seed(seed=seed)
    if mc_cov is None:
        toys = np.random.poisson(exp, size=(toys_size, len(exp)))
    else:
        # toys = np.random.multivariate_normal(mean=exp, cov=mc_cov, size=(toys_size, len(exp)))
        toys_base = np.random.lognormal(mean=exp, sigma=np.sqrt(np.diagonal(mc_cov)), size=(toys_size, len(exp)))
        toys = np.random.poisson(lam=toys_base)

    toy_chi_squared = calc_chi_squared(toys, exp, exp_unc)

    return obs_chi_squared, toy_chi_squared


def toy_chi2_test(data, expectation, error, mc_cov=None, toys_size=1000000):
    """
    Performs a GoF-test using a test statistic based on toy MC sampled
    from the expected distribution.

    Parameters
    ----------
    data : np.ndarray
        Data bin counts. Shape is (`num_bins`,)
    expectation : np.ndarray
        Expected bin counts. Shape is (`num_bins`,)
    error : np.ndarray
        Uncertainty on the expected distribution. Shape is (`num_bins`,)
    toys_size : int
        Number of toy samples to be drawn from expectation to model the chi2
        of the expectation. Default is 1000000.

    Returns
    -------
    float
        :math:`\chi^2`
    float
        p-value.
    tuple(bin_counts, bin_edges, chi2_toys)
        Information needed to plot the chi2 distribution obtained from the toys.
    """
    obs_chi2, toys = mc_chi_squared_from_toys(obs=data, exp=expectation, exp_unc=error, mc_cov=mc_cov, toys_size=toys_size)

    bc, be = np.histogram(toys, bins=100, density=True)

    bm = (be[1:] + be[:-1]) / 2
    bw = (be[1:] - be[:-1])
    p_val = np.sum(bc[bm > obs_chi2] * bw[bm > obs_chi2])

    return obs_chi2, p_val, (bc, be, toys)
