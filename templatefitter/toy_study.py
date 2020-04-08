import tqdm
import logging
import numpy as np

from typing import Tuple, List, Dict, Any

from templatefitter.fitter import TemplateFitter
from templatefitter.fit_model.model_builder import FitModel

__all__ = [
    "ToyStudy"
]

logging.getLogger(__name__).addHandler(logging.NullHandler())


# TODO!!!!!


class ToyStudy:
    """
    This class helps you to perform toy monte carlo studies
    using given templates and an implementation of a negative
    log likelihood function. This is useful to discover possible
    biases or a over/under estimation of errors for fit parameters.

    Parameters
    ----------
    fit_model : FitModel
        A instance of the FitModel class.
    minimizer_id : str
        A string specifying the method to be used for  the
        minimization of the Likelihood function. Available are
        'scipy' and 'iminuit'.

    """

    def __init__(self, fit_model: FitModel, minimizer_id: str) -> None:
        self._fit_model = fit_model
        self._minimizer_id = minimizer_id
        self._toy_results = {"parameters": [], "uncertainties": []}  # type: Dict[str, List[Any]]
        self._is_fitted = False  # type: bool

    def do_experiments(self, n_exp: int = 1000, max_tries: int = 10) -> None:
        """
        Performs fits using the given template and generated
        toy monte carlo (following a poisson distribution) as data.

        Parameters
        ----------
        n_exp : int
            Number of toy experiments to run.
        max_tries : int
            Maximum number of tries for an experiment if a RuntimeError
            occurs.
        """
        self._reset_state()

        logging.info(f"Performing toy study with {n_exp} experiments...")

        for _ in tqdm.tqdm(range(n_exp), desc="Experiments Progress"):
            self._experiment(max_tries)

        self._is_fitted = True

    def _experiment(self, max_tries: int = 10, get_hesse: bool = True) -> None:
        """
        Helper function for toy experiments.
        """
        for _ in range(max_tries):
            try:

                # TODO: Templates.generate_toy_dataset() does not exist in this version, yet!
                self._templates.add_data(**self._templates.generate_toy_dataset())

                fitter = TemplateFitter(
                    self._templates, minimizer_id=self._minimizer_id
                )
                result = fitter.do_fit(update_templates=False,
                                       verbose=False,
                                       get_hesse=get_hesse,
                                       fix_nui_params=True)

                self._toy_results["parameters"].append(result.params.values)
                self._toy_results["uncertainties"].append(result.params.errors)

                return

            except RuntimeError:
                logging.debug("RuntimeError occurred in toy experiment. Trying again")
                continue

        raise RuntimeError("Experiment exceed max number of retries.")

    # TODO
    def do_background_linearity_test(
            self,
            signal_id: str,
            background_id: str,
            limits: Tuple[float, float],
            n_points: int = 10,
            n_exp: int = 200
    ) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        Parameters
        ----------
        signal_id : str
            Name of the template for which the linearity test
            should be performed.
        background_id : str
            Name of the template which is the background to the template
            for which the linearity test should be performed.
        limits : tuple of float
            Range where the yield parameter will be tested in.
        n_points : int, optional
            Number of points to test in the given range. This
            samples `n_points` in a linear space in the range
            specified by `limits`. Default is 10.
        n_exp : int, optional
            Number of toy experiments to perform per point.
            Default is 100.
        """
        param_fit_results = list()  # type: List[float]
        param_fit_errors = list()  # type: List[float]
        param_points = np.linspace(start=limits[0], stop=limits[1], num=n_points)  # type: np.ndarray
        assert isinstance(param_points, np.ndarray), type(param_points)

        logging.info(f"Performing linearity test for parameter: {signal_id}")

        for param_point in tqdm.tqdm(param_points, desc="Linearity Test Progress"):
            self._reset_state()
            self._templates.reset_parameters()

            self._templates.set_yield(background_id, param_point)

            for _ in tqdm.tqdm(range(n_exp), desc="Experiment Progress"):
                self._experiment(get_hesse=False)

            self._is_fitted = True

            params, _ = self.get_toy_results(signal_id)
            param_fit_results.append(float(np.mean(params)))
            param_fit_errors.append(float(np.std(params)))

            assert all(isinstance(fr, float) for fr in param_fit_results)
            assert all(isinstance(fe, float) for fe in param_fit_errors)

        return param_points, param_fit_results, param_fit_errors

    def do_linearity_test(
            self,
            process_id: str,
            limits: Tuple[float, float],
            n_points: int = 10,
            n_exp: int = 200
    ) -> Tuple[np.ndarray, List[float], List[float]]:

        """
        Performs a linearity test for the yield parameter of
        the specified template.

        Parameters
        ----------
        process_id : str
            Name of the template for which the linearity test
            should be performed.
        limits : tuple of float
            Range where the yield parameter will be tested in.
        n_points : int, optional
            Number of points to test in the given range. This
            samples `n_points` in a linear space in the range
            specified by `limits`. Default is 10.
        n_exp : int, optional
            Number of toy experiments to perform per point.
            Default is 200.
        """
        return self.do_background_linearity_test(
            signal_id=process_id,
            background_id=process_id,
            limits=limits,
            n_points=n_points,
            n_exp=n_exp
        )

    @property
    def result_parameters(self) -> np.ndarray:
        """
        np.ndarray: A 2D array of fit results for the parameters
        of the likelihood.
        """
        self._check_state()
        return np.array(self._toy_results["parameters"])

    @property
    def result_uncertainties(self) -> np.ndarray:
        """
        np.ndarray: A 2D array of uncertainties fo the fit
        results for the parameters of the likelihood.
        """
        self._check_state()
        return np.array(self._toy_results["uncertainties"])

    def get_toy_results(self, process_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns results from the toy Monte Carlo study.

        Parameters
        ----------
        process_id : int, list of int
            Index or indices of the parameter of interest.

        Returns
        -------
        parameters : np.ndarray
            Results for the fitted values of parameters specified by
            `param_index`. Shape is (`n_exp`, `len(param_index)`).
        uncertainties : np.ndarray
            Results for the uncertainties of fitted values for parameters
            specified by `param_index`. Shape is (`n_exp`, `len(param_index)`).
        """
        self._check_state()
        process_id = self._templates.process_to_index(process_id)
        parameters = self.result_parameters[:, process_id]
        uncertainties = self.result_uncertainties[:, process_id]

        assert len(parameters.shape) == 1, parameters.shape
        assert len(uncertainties.shape) == 1, uncertainties.shape

        return parameters, uncertainties

    def get_toy_result_pulls(self, process_id: str) -> np.ndarray:
        """
        Returns pulls of the results from the toy Monte Carlo study. The pull is defined as

        :math:`p=\\frac{\\nu^{\mathrm{fit}} - \\nu^{\mathrm{exp}}}{\sigma_{\\nu^{\mathrm{exp}}}}`,

        and should follow a standard normal distribution.

        Parameters
        ----------
        process_id : int, list of int
            Index or indices of the parameter of interest.

        Returns
        -------
        pulls : np.ndarray
            Pull values for the fitted values of parameters specified by
            `param_index`. Shape is (`n_exp`, `len(param_index)`).
        """

        self._check_state()
        parameters, uncertainties = self.get_toy_results(process_id)
        # TODO: this works only for template yield, for nuisance parameters I have to change this
        expected_yield = self._templates.get_yield(process_id)

        return (parameters - expected_yield) / uncertainties

    def _check_state(self) -> None:
        """
        Checks the state of the class instance. If no toy
        experiments have been performed, a RuntimeError will
        be raised.

        Raises
        ------
        RuntimeError
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Toy experiments have not yet been performed. "
                " Execute 'do_experiments' first."
            )

    def _reset_state(self) -> None:
        """
        Resets state of the ToyStudy. This removes the toy results
        and set the state to not fitted.
        """
        self._is_fitted = False
        self._toy_results["parameters"] = list()
        self._toy_results["uncertainties"] = list()
