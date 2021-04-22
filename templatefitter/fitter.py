import tqdm
import logging
import numpy as np

from multiprocessing import Pool
from typing import Union, Tuple, List, Dict

from templatefitter.fit_model.model_builder import FitModel
from templatefitter.minimizer import (
    AbstractMinimizer,
    available_template_fitter_minimizer,
    BoundType,
    minimizer_factory,
    MinimizeResult,
)


__all__ = [
    "TemplateFitter",
]


logging.getLogger(__name__).addHandler(logging.NullHandler())


class TemplateFitter:
    """
    This class performs the parameter estimation and calculation
    of a profile likelihood based on a constructed negative log
    likelihood function.

    Parameters
    ----------
    fit_model : Implemented FitModel
        An instance of a FitModel class that provides a negative log likelihood function via the `create_nll` method.
    minimizer_id : str
        A string specifying the method to be used for the minimization of the Likelihood function.
        Available are 'scipy' and 'iminuit'.
    """

    def __init__(
        self,
        fit_model: FitModel,
        minimizer_id: str,
    ) -> None:
        if minimizer_id not in available_template_fitter_minimizer:
            raise KeyError(
                f"The parameter 'minimizer_id' defining the Minimizer to be used must be on of\n"
                f"{list(available_template_fitter_minimizer.keys())}\n"
                f"You provided: minimizer_id = {minimizer_id}"
            )

        self._fit_model = fit_model
        self._nll = self._fit_model.create_nll()
        self._nll_creator = self._fit_model.create_nll
        self._minimizer_id = minimizer_id

        self._fit_result = None
        self._fixed_parameters = list()  # type: List[Union[str, int]]
        self._bound_parameters = dict()  # type: Dict[Union[str, int], BoundType]

    def do_fit(
        self,
        update_templates: bool = True,
        get_hesse: bool = True,
        verbose: bool = True,
        fix_nui_params: bool = False,
    ) -> MinimizeResult:
        """
        Performs maximum likelihood fit by minimizing the provided negative log likelihood function.

        Parameters
        ----------
        update_templates : bool, optional
            Whether to update the parameters of the given will_templates
            or not. Default is True.
        verbose : bool, optional
            Whether to print fit information or not. Default is True
        fix_nui_params : bool, optional
            Whether to fix nuisance parameters in the fit or not.
            Default is False.
        get_hesse : bool, optional
            Whether to calculate the Hesse matrix in the estimated
            minimum of the negative log likelihood function or not.
            Can be computationally expensive if the number of parameters
            in the likelihood is high. It is only needed for the scipy
            minimization method. Default is True.

        Returns
        -------
        MinimizeResult : namedtuple
            A namedtuple with the most important information about the minimization.
        """
        minimizer = minimizer_factory(
            minimizer_id=self._minimizer_id,
            fcn=self._nll_creator(fix_nuisance_parameters=fix_nui_params),
            names=self._nll.param_names,
        )

        if fix_nui_params:
            for param_id in self._fit_model.floating_nuisance_parameter_indices:
                minimizer.set_param_fixed(param_id=param_id)

        for param_id_or_str in self._fixed_parameters:
            minimizer.set_param_fixed(param_id=param_id_or_str)

        for param_id_or_str, bounds in self._bound_parameters.items():
            minimizer.set_param_bounds(param_id=param_id_or_str, bounds=bounds)

        fit_result = minimizer.minimize(
            initial_param_values=self._nll.x0,
            verbose=verbose,
            get_hesse=get_hesse,
        )

        if update_templates:
            self._fit_model.update_parameters(parameter_vector=fit_result.params.values)

        return fit_result

    def set_parameter_fixed(
        self,
        param_id: Union[int, str],
    ) -> None:
        """
        Adds parameter to the fixed parameter list.

        Parameters
        ----------
        param_id : str or int
            Parameter identifier.
        """
        self._fixed_parameters.append(param_id)

    def set_parameter_bounds(
        self,
        param_id: Union[str, int],
        bounds: BoundType,
    ) -> None:
        """
        Adds parameter and its boundaries to the bound  parameter dictionary.

        Parameters
        ----------
        param_id : str or int
            Parameter identifier.
        bounds : tuple of float
            Lower and upper boundaries for this parameter.
        """

        self._bound_parameters[param_id] = bounds

    @staticmethod
    def _get_hesse_approx(
        param_id: Union[int, str],
        fit_result: MinimizeResult,
        profile_points: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates a gaussian approximation of the negative log likelihood function using the Hesse matrix.

        Parameters
        ----------
        param_id : int or string
            Parameter index or name.
        fit_result : MinimizeResult
            A namedtuple with the most important information about the minimization.
        profile_points : np.ndarray
            Points where the estimate is evaluated. Shape is (num_points,).

        Returns
        -------
        np.ndarray
            Hesse approximation of the negative log likelihood function. Shape is (num_points,).

        """

        result = fit_result.params.get_param_value(param_id=param_id)
        param_index = fit_result.params.param_id_to_index(param_id=param_id)
        hesse_error = fit_result.params.errors[param_index]
        hesse_approx = 0.5 * (1 / hesse_error) ** 2 * (profile_points - result) ** 2 + fit_result.fcn_min_val

        return hesse_approx

    def profile(
        self,
        param_id: Union[int, str],
        num_cpu: int = 4,
        num_points: int = 100,
        sigma: float = 2.0,
        subtract_min: bool = True,
        fix_nui_params: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs a profile scan of the negative log likelihood function for the specified parameter.

        Parameters
        ----------
        param_id : int or string
            Parameter index or name.
        num_cpu : int
            Maximal number of processes to uses.
        num_points : int
            Number of points where the negative log likelihood is
            minimized.
        sigma : float
            Defines the width of the scan. The scan range is given by sigma * uncertainty of the given parameter.
        subtract_min : bool, optional
            Whether to subtract the estimated minimum of the negative log likelihood function or not. Default is True.
        fix_nui_params : bool, optional
            Whether to fix nuisance parameters. Default is False.

        Returns
        -------
        np.ndarray
            Scan points. Shape is (num_points,).
        np.ndarray
            Profile values. Shape is (num_points,).
        np.ndarray
            Hesse approximation. Shape is (num_points,).
        """
        logging.info(f"\nCalculating profile likelihood for parameter: '{param_id}'")

        minimizer = minimizer_factory(
            minimizer_id=self._minimizer_id,
            fcn=self._nll_creator(fix_nuisance_parameters=fix_nui_params),
            names=self._nll.param_names,
        )

        if fix_nui_params:
            for param_id in self._fit_model.floating_nuisance_parameter_indices:
                minimizer.set_param_fixed(param_id=param_id)

        for fix_param_id in self._fixed_parameters:
            minimizer.set_param_fixed(fix_param_id)

        logging.info("Start nominal minimization")
        result = minimizer.minimize(initial_param_values=self._nll.x0, get_hesse=True, verbose=True)

        minimum = result.fcn_min_val
        param_val, param_unc = minimizer.params[param_id]

        profile_points = np.linspace(param_val - sigma * param_unc, param_val + sigma * param_unc, num_points)

        hesse_approx = self._get_hesse_approx(param_id=param_id, fit_result=result, profile_points=profile_points)

        logging.info(f"Start profiling the likelihood using {num_cpu} processes...")
        args = [(minimizer, point, result.params.values, param_id, fix_nui_params) for point in profile_points]
        with Pool(num_cpu) as pool:
            profile_values = np.array(
                list(
                    tqdm.tqdm(
                        pool.imap(self._profile_helper, args),
                        total=len(profile_points),
                        desc="Profile Progress",
                    )
                )
            )

        if subtract_min:
            profile_values -= minimum
            hesse_approx -= minimum

        return profile_points, profile_values, hesse_approx

    def _profile_helper(
        self,
        args: Tuple[AbstractMinimizer, float, np.ndarray, Union[int, str], bool],
    ) -> float:
        """
        Helper function for the calculation for the profile nll.

        Parameters
        ----------
        args: tuple
            1st element: Minimizer object,
            2nd element: Parameter point,
            3rd element: Initial parameter values,
            4th element: Parameter identifier.
            5th element: Boolean indicator for fixing nuisance parameters.

        Returns
        -------
        fcn_min_val : float
            Minimum function value.
        """

        minimizer = args[0]
        point = args[1]
        initial_values = args[2]
        param_id = args[3]
        fix_nui_params = args[4]

        minimizer.release_params()
        param_index = minimizer.params.param_id_to_index(param_id=param_id)
        initial_values[param_index] = point
        if fix_nui_params:
            for param_id in self._fit_model.floating_nuisance_parameter_indices:
                minimizer.set_param_fixed(param_id=param_id)

        minimizer.set_param_fixed(param_id)
        for param_id in self._fixed_parameters:
            minimizer.set_param_fixed(param_id=param_id)

        try:
            loop_result = minimizer.minimize(initial_param_values=initial_values, get_hesse=False)
        except RuntimeError as e:
            logging.info(e)
            logging.info(f"Minimization with point {point} was not successful, trying again.")
            return np.nan

        return loop_result.fcn_min_val

    def get_significance(
        self,
        yield_parameter: str,
        verbose: bool = True,
        fix_nui_params: bool = False,
    ) -> float:
        """
        Calculate significance for the specified yield parameter using the profile likelihood ratio.

        The significance is base on the profile likelihood ratio

            lambda(nu) = L(nu, hat(hat(theta))) / L(hat(nu), hat(theta)),

        where hathat(theta) maximizes L for a specified nu and (hat(nu), hat(theta}) maximizes L totally.

        The test statistic used for discovery is

            q_0 =
            -2 / log(lambda(0))     for hat(nu) >= 0, or
            0                       for hat(nu) < 0

        Parameters
        ----------
        yield_parameter : str
            Name of yield_parameter of a fit component of the FitModel
            for which the significance should be calculated.
        verbose : bool, optional
            Whether to show output. Default is True.
        fix_nui_params : bool, optional
            Whether to fix nuisance parameters. Default is False.

        Returns
        -------
        significance : float
            Fit significance for the yield parameter in gaussian
            standard deviations.
        """

        # perform the nominal minimization
        minimizer = minimizer_factory(
            minimizer_id=self._minimizer_id,
            fcn=self._nll_creator(fix_nuisance_parameters=fix_nui_params),
            names=self._nll.param_names,
        )

        if fix_nui_params:
            for param_id in self._fit_model.floating_nuisance_parameter_indices:
                minimizer.set_param_fixed(param_id=param_id)

        logging.info("Perform nominal minimization:")
        for param_id_or_str in self._fixed_parameters:
            minimizer.set_param_fixed(param_id=param_id_or_str)

        for param_id_or_str, bounds in self._bound_parameters.items():
            minimizer.set_param_bounds(param_id=param_id_or_str, bounds=bounds)
        fit_result = minimizer.minimize(initial_param_values=self._nll.x0, verbose=verbose)

        if fit_result.params[yield_parameter][0] < 0:
            return 0.0

        # set signal of template related to the specified yield_parameter to zero and profile the likelihood
        self._fit_model.set_initial_parameter_value(parameter_name=yield_parameter, new_initial_value=0.0)
        self._fit_model.reset_parameters_to_initial_values()

        minimizer_bkg = minimizer_factory(
            minimizer_id=self._minimizer_id,
            fcn=self._nll_creator(fix_nuisance_parameters=fix_nui_params),
            names=self._nll.param_names,
        )

        if fix_nui_params:
            for param_id in self._fit_model.floating_nuisance_parameter_indices:
                minimizer_bkg.set_param_fixed(param_id=param_id)

        for param_id_or_str in self._fixed_parameters:
            minimizer_bkg.set_param_fixed(param_id=param_id_or_str)

        for param_id_or_str, bounds in self._bound_parameters.items():
            minimizer_bkg.set_param_bounds(param_id=param_id_or_str, bounds=bounds)

        minimizer_bkg.set_param_fixed(param_id=yield_parameter)
        logging.info("Background")
        profile_result = minimizer_bkg.minimize(initial_param_values=self._nll.x0, verbose=verbose)

        assert profile_result.params[yield_parameter][0] == 0.0, profile_result.params[yield_parameter][0]

        q0 = 2 * (profile_result.fcn_min_val - fit_result.fcn_min_val)

        logging.debug(
            f"For yield_parameter {yield_parameter}: q0: {q0}, {profile_result.params[yield_parameter][0]}, {fit_result.params[yield_parameter][0]}, "
            f"{profile_result.fcn_min_val}, {fit_result.fcn_min_val}"
        )
        assert q0 >= 0.0, (q0, yield_parameter)

        self._fit_model.reset_initial_parameter_value(parameter_name=yield_parameter)

        return np.sqrt(q0)
