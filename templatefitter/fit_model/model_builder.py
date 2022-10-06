"""
Class which defines the fit model by combining templates and handles the computation.
"""

import copy
import logging
import operator
import numpy as np
import scipy.stats as scipy_stats

from numba import jit
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, Sequence


from templatefitter.utility import xlogyx, cov2corr

from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.binned_distribution import DataInputType

from templatefitter.fit_model.template import Template
from templatefitter.fit_model.component import Component
from templatefitter.fit_model.data_channel import ModelDataChannels
from templatefitter.fit_model.channel import ModelChannels, Channel
from templatefitter.fit_model.constraint import Constraint, ConstraintContainer
from templatefitter.fit_model.parameter_handler import ParameterHandler, ModelParameter, TemplateParameter
from templatefitter.fit_model.utility import pad_sequences, check_bin_count_shape, immutable_cached_property
from templatefitter.fit_model.fit_object_managers import (
    FractionConversionInfo,
    FractionManager,
    FitObjectManager,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "FitModel",
]


# TODO: Not yet considered are Yield Ratios:
#       We could add a ratio parameter instead of a yield parameter for components for which
#       the ratio of two yields should be fitted.
#       This would require an additional parameter type and an additional vector in the
#       calculation in calculate_expected_bin_count:
#           yield_vector * ratio_vector
#       where ratio vector holds ones except for the row of the component which is related to another component
#       via the ratio. For this component the respective yield must stand at the place of its own yield value, e.g.:
#           ratio = yield_2 / yield_1 -> yield_2 = ratio * yield_1
#           => (yield_i, yield_1, yield_1, yield_j, ...)^T * (1, 1, ratio, 1, ...)^T


# TODO: Maybe the FitModel could produce a Model object, which is a container that holds all
#       the necessary information and can be used to recreate the model.


# TODO next: - Include uncertainties into yield or template shapes!
#            - uncertainties must be relative uncertainties!
#            - generalize uncertainty handling!


class FitModel:
    def __init__(
        self,
        parameter_handler: ParameterHandler,
        ignore_empty_mc_bins: bool = False,
    ):

        # region private attributes
        self._params = parameter_handler  # type: ParameterHandler

        self._ignore_empty_mc_bins = ignore_empty_mc_bins  # type: bool

        self._model_parameters = []  # type: List[ModelParameter]
        self._model_parameters_mapping = {}  # type: Dict[str, int]

        self._template_manager: FitObjectManager[Template] = FitObjectManager()
        self._component_manager: FitObjectManager[Component] = FitObjectManager()

        self._channels = ModelChannels()  # type: ModelChannels

        self._data_channels = ModelDataChannels()  # type: ModelDataChannels
        self._original_data_channels = None  # type: Optional[ModelDataChannels]

        self._masked_data_bin_counts = None  # type: Optional[np.ndarray]
        self._data_stat_errors_sq = None  # type: Optional[np.ndarray]

        self._fraction_manager = FractionManager(
            param_handler=self._params,
            channels=self._channels,
        )  # type: FractionManager

        self._inverse_template_bin_correlation_matrix = None  # type: Optional[np.ndarray]
        self._systematics_covariance_matrices_per_channel = None  # type: Optional[List[np.ndarray]]

        self._has_data = False  # type: bool
        self._is_initialized = False  # type: bool
        self._is_checked = False  # type: bool

        self._yield_indices = None  # type: Optional[List[int]]
        self._yields_checked = False  # type: bool

        self._fraction_indices = None  # type: Optional[List[int]]
        self._fractions_checked = False  # type: bool

        self._efficiency_indices = None  # type: Optional[List[int]]
        self._efficiency_reshaping_indices = None  # type: Optional[List[int]]
        self._efficiency_padding_required = True  # type: bool
        self._efficiencies_checked = False  # type: bool

        self._bin_nuisance_params_checked = False  # type: bool

        self._constraint_container = ConstraintContainer()  # type: ConstraintContainer

        self._template_shapes_checked = False  # type: bool
        self._template_shape = None  # type: Optional[np.ndarray]

        self._gauss_term_checked = False  # type: bool
        self._constraint_term_checked = False  # type: bool
        self._chi2_calculation_checked = False  # type: bool
        self._nll_calculation_checked = False  # type: bool

        # endregion

        # Setting a random seed for the toy data set generation with SciPy
        self._random_state = np.random.RandomState(seed=7694747)  # type: np.random.RandomState

    # region Basic Properties
    # Attributes forwarded to self._channels via __getattr__()
    _channel_attrs = [
        "binning",
        "max_number_of_bins_flattened",
        "min_number_of_independent_yields",
        "number_of_bins_flattened_per_channel",
        "number_of_components",
        "number_of_dependent_templates",
        "number_of_expected_independent_yields",
        "number_of_fraction_parameters",
        "number_of_independent_templates",
        "number_of_templates",
        "template_bin_counts",
        "total_number_of_templates",
    ]

    def __getattribute__(self, attr):
        """
        Forwarding some attributes to self._channels for backwards compatibility.
        :param attr: The attribute name
        :return: A forwarded attribute of ModelChannels
        """

        try:
            return super().__getattribute__(attr)
        except AttributeError:
            if attr in self._channel_attrs:
                try:
                    return getattr(self._channels, attr)
                except AttributeError:
                    raise AttributeError(
                        f"Forwarded attribute access from {self.__class__.__name__} to"
                        f" {self._channels.__class__.__name__} which also has no attribute {attr}."
                    )
            elif attr in dir(self._channels):
                raise AttributeError(
                    f"Attribute {attr} exists for {self._channels.__class__.__name__} but is not forwarded "
                    f"to {self.__class__.__name__}."
                )
            else:
                raise

    @property
    def number_of_channels(self) -> int:
        return len(self._channels)

    @property
    def fraction_conversion(self) -> FractionConversionInfo:
        assert self._fraction_manager.fraction_conversion is not None
        return self._fraction_manager.fraction_conversion

    @property
    def is_finalized(self) -> bool:
        return self._is_initialized

    # endregion

    # region Add parameters, templates, components, channels, constraints and data

    def add_model_parameter(
        self,
        name: str,
        parameter_type: str,
        floating: bool,
        initial_value: float,
        constrain_to_value: Optional[float] = None,
        constraint_sigma: Optional[float] = None,
    ) -> Tuple[int, ModelParameter]:
        self._check_is_not_finalized()
        self._check_has_data(adding="model parameter")

        if name in self._model_parameters_mapping:
            raise RuntimeError(
                f"The model parameter with the name {name} already exists!\n"
                f"It has the following properties:\n"
                f"{self._model_parameters[self._model_parameters_mapping[name]]}"
            )

        model_index = len(self._model_parameters)
        model_parameter = ModelParameter(
            name=name,
            parameter_handler=self._params,
            parameter_type=parameter_type,
            model_index=model_index,
            floating=floating,
            initial_value=initial_value,
            constrain_to_value=constrain_to_value,
            constraint_sigma=constraint_sigma,
        )
        self._model_parameters.append(model_parameter)
        self._model_parameters_mapping.update({name: model_index})
        return model_index, model_parameter

    def add_template(
        self,
        template: Template,
        yield_parameter: Union[ModelParameter, str],
        use_other_systematics: bool = True,
    ) -> int:
        self._check_is_not_finalized()
        self._check_has_data(adding="template")

        self._template_manager.append(template)

        if isinstance(yield_parameter, str):
            yield_model_parameter = self.get_model_parameter(name_or_index=yield_parameter)
        elif isinstance(yield_parameter, ModelParameter):
            self._check_model_parameter_registration(model_parameter=yield_parameter)
            yield_model_parameter = yield_parameter
        else:
            raise ValueError(
                f"Expected to receive object of type string or ModelParameter "
                f"for argument yield_parameter, but you provided object of type {type(yield_parameter)}!"
            )

        if not yield_model_parameter.parameter_type == ParameterHandler.yield_parameter_type:
            raise ValueError(
                f"The ModelParameter provided for the template yield must be of parameter_type 'yield', "
                f"however, the ModelParameter you provided is of parameter_type "
                f"'{yield_model_parameter.parameter_type}'"
            )

        yield_param = TemplateParameter(
            name=f"{template.name}_{yield_model_parameter.name}",
            parameter_handler=self._params,
            parameter_type=yield_model_parameter.parameter_type,
            floating=yield_model_parameter.floating,
            initial_value=yield_model_parameter.initial_value,
            param_id=yield_model_parameter.param_id,
        )

        yield_model_parameter.used_by(template_parameter=yield_param, template_serial_number=template.serial_number)

        bin_nuisance_paras = self._create_bin_nuisance_parameters(template=template)
        assert len(bin_nuisance_paras) == template.num_bins_total, (len(bin_nuisance_paras), template.num_bins_total)

        template.initialize_parameters(
            yield_parameter=yield_param,
            bin_nuisance_parameters=bin_nuisance_paras,
        )

        template.use_other_systematics = use_other_systematics

        return template.serial_number

    @staticmethod
    def _validate_add_component_arguments(
        component: Optional[Component] = None,
        name: Optional[str] = None,
        templates: Optional[List[Union[int, str, Template]]] = None,
        shared_yield: Optional[bool] = None,
    ) -> None:

        component_input_error_text = (
            "You can either add an already prepared component or create a new one.\n"
            "For the first option, the argument 'component' must be used;\n"
            "For the second option, please use the arguments 'templates', 'name' and "
            "'shared_yield' as you would when creating a new Component object.\n"
            "In the latter case, the templates can also be provided via their name or "
            "serial_number es defined for this model."
        )

        if (component is None) == (templates is None):
            raise ValueError(component_input_error_text)
        elif component is not None:
            if not isinstance(component, Component):
                raise ValueError(
                    f"The argument 'component' must be of type 'Component', "
                    f"but you provided an object of type {type(component)}"
                )
            if name is not None and name != component.name:
                raise ValueError(
                    f"You set the argument 'name' to {name} despite the provided component "
                    f"already having a different name, which is {component.name}"
                )
            if shared_yield is not None and (shared_yield != component.shared_yield):
                raise ValueError(
                    f"You set the argument 'shared_yield' to {shared_yield} despite the provided"
                    f" component already having shared_yield set to {component.shared_yield}."
                )

        elif templates is not None:
            if not (
                isinstance(templates, list)
                and all(isinstance(t, Template) or isinstance(t, int) or isinstance(t, str) for t in templates)
            ):
                raise ValueError(
                    "The argument 'templates 'takes a list of Templates, integers or strings, but you "
                    "provided "
                    + (
                        f"an object of type {type(templates)}"
                        if not isinstance(templates, list)
                        else f"a list containing the types {[type(t) for t in templates]}"
                    )
                )
            if shared_yield is None and len(templates) > 1:
                raise ValueError(
                    "If you want to directly create and add a component, you have to specify whether the "
                    "templates of the component shall share their yields via the boolean parameter "
                    "'shared_yields'!"
                )
            if name is None:
                raise ValueError(
                    "If you want to directly create and add component, you have to specify its name "
                    "via the argument 'name'!"
                )

    def add_component(
        self,
        fraction_parameters: Optional[List[Union[ModelParameter, str]]] = None,
        component: Optional[Component] = None,
        name: Optional[str] = None,
        templates: Optional[List[Union[int, str, Template]]] = None,
        shared_yield: Optional[bool] = None,
    ) -> Union[int, Tuple[int, Component]]:

        self._check_is_not_finalized()
        self._check_has_data(adding="component")

        self._validate_add_component_arguments(component, name, templates, shared_yield)

        if templates is not None:
            template_list = []
            for template in templates:
                if isinstance(template, Template):
                    template_list.append(template)
                elif isinstance(template, int) or isinstance(template, str):
                    template_list.append(self.get_template(name_or_index=template))
                else:
                    raise ValueError(
                        f"Parameter 'templates' must be a list of strings, integer or Templates "
                        f"to allow for the creation of a new component.\n"
                        f"You provided a list containing the types {[type(t) for t in templates]}."
                    )
            assert name is not None and shared_yield is not None
            component = Component(
                templates=template_list,
                params=self._params,
                name=name,
                shared_yield=shared_yield,
            )

        assert component is not None

        if component.required_fraction_parameters == 0:
            if fraction_parameters:
                raise ValueError(
                    f"The component requires no fraction parameters, but you provided {len(fraction_parameters)}!"
                )
            component.initialize_parameters(fraction_parameters=None)

        else:
            assert fraction_parameters is not None
            if not all(isinstance(p, ModelParameter) or isinstance(p, str) for p in fraction_parameters):
                raise ValueError(
                    f"Expected to receive list of string or ModelParameters for argument "
                    f"fraction_parameters, but you provided a list containing objects of the types "
                    f"{[type(p) for p in fraction_parameters]}!"
                )

            fraction_params = self._get_list_of_template_params(
                input_params=fraction_parameters,
                serial_numbers=component.template_serial_numbers,
                container_name=component.name,
                parameter_type=ParameterHandler.fraction_parameter_type,
                input_parameter_list_name="fraction_parameters",
            )
            component.initialize_parameters(fraction_parameters=fraction_params)

        self._component_manager.append(component)

        if templates is not None:
            return component.component_serial_number, component
        else:
            return component.component_serial_number

    @staticmethod
    def _validate_add_channel_arguments(
        channel: Optional[Channel] = None,
        name: Optional[str] = None,
        components: Optional[Sequence[Union[int, str, Component, Template]]] = None,
    ) -> None:

        input_error_text = (
            "A channel can either be added by providing\n"
            "\t- an already prepared channel via the argument 'channel'\nor\n"
            "\t- a list of components (directly, via their names or via their serial_numbers)\n"
            "\t    (or alternatively Templates or template serial_numbers or names with a prefix "
            "\t    'temp_', from which a single template component will be generated)"
            "\t  and a name for the channel, using the arguments 'components' and 'name', respectively."
        )

        if channel is not None and components is not None:
            raise ValueError(input_error_text)
        elif channel is not None:
            if not isinstance(channel, Channel):
                raise ValueError(
                    f"The argument 'channel' must be of type 'Channel', "
                    f"but you provided an object of type {type(channel)}"
                )
            if name is not None and name != channel.name:
                raise ValueError(
                    f"You set the argument 'name' to {name} despite the provided channel "
                    f"already having a different name, which is {channel.name}"
                )
        elif components is not None:
            if not (
                isinstance(components, list)
                and all(
                    isinstance(c, Component) or isinstance(c, int) or isinstance(c, str) or isinstance(c, Template)
                    for c in components
                )
            ):
                raise ValueError(
                    "The argument 'components 'takes a list of Components, integers, strings or Templates,"
                    " but you provided " + f"an object of type {type(components)}"
                    if not isinstance(components, list)
                    else f"a list containing the types {[type(c) for c in components]}"
                )
            if name is None:
                raise ValueError("When directly creating and adding a channel, you have to set the argument 'name'!")

    def add_channel(
        self,
        efficiency_parameters: List[Union[ModelParameter, str]],
        channel: Optional[Channel] = None,
        name: Optional[str] = None,
        components: Optional[Sequence[Union[int, str, Component, Template]]] = None,
        latex_label: Optional[str] = None,
        plot_order: Optional[Tuple[str, ...]] = None,
    ) -> Union[int, Tuple[int, Channel]]:

        self._check_is_not_finalized()

        self._validate_add_channel_arguments(channel, name, components)
        self._check_has_data(adding="channel")

        if components is not None:
            component_list = []  # type: List[Component]
            for component in components:
                if isinstance(component, Component):
                    component_list.append(component)
                if isinstance(component, Template):
                    new_component = self._create_single_template_component_from_template(template_input=component)
                    component_list.append(new_component)
                elif isinstance(component, int) or isinstance(component, str):
                    if isinstance(component, str) and component.startswith("temp_"):
                        # Handling case where a template identifier (string or index with prefix 'temp_') was provided:
                        temp_name_or_index_str = component[len("temp_") :]
                        new_component = self._create_single_template_component_from_template(
                            template_input=temp_name_or_index_str
                        )
                        component_list.append(new_component)
                    else:
                        component_list.append(self.get_component(name_or_index=component))
                else:
                    raise ValueError(f"Unexpected type {type(component)} for element of provided list of components.")

            assert name is not None
            channel = Channel(
                params=self._params,
                name=name,
                components=component_list,
                latex_label=latex_label,
                plot_order=plot_order,
            )

        assert channel is not None
        if len(efficiency_parameters) != channel.required_efficiency_parameters:
            raise ValueError(
                f"The channel requires {channel.required_efficiency_parameters} efficiency parameters, "
                f"but you provided {len(efficiency_parameters)}!"
            )

        efficiency_params = self._get_list_of_template_params(
            input_params=efficiency_parameters,
            serial_numbers=channel.template_serial_numbers,
            container_name=channel.name,
            parameter_type=ParameterHandler.efficiency_parameter_type,
            input_parameter_list_name="efficiency_parameters",
        )

        channel.initialize_parameters(efficiency_parameters=efficiency_params)
        channel_serial_number = self._channels.add_channel(channel=channel)

        if components is not None:
            return channel_serial_number, channel
        else:
            return channel_serial_number

    def add_constraint(
        self,
        name: str,
        value: float,
        sigma: float,
    ) -> None:
        self._check_is_not_finalized()

        if name not in self._model_parameters_mapping:
            raise ValueError(
                f"A ModelParameter with the name '{name}' was not added, yet, "
                f"and thus a constraint cannot be applied to it!"
            )

        model_parameter = self._model_parameters[self._model_parameters_mapping[name]]
        if model_parameter.constraint_value is not None:
            raise RuntimeError(
                f"The ModelParameter '{name}' already is constrained with the settings"
                f"\n\tconstraint_value = {model_parameter.constraint_value}"
                f"\n\tconstraint_sigma = {model_parameter.constraint_sigma}\n"
                f"and thus your constraint (cv = {value}, cs = {sigma}) cannot be applied!"
            )

        assert model_parameter.param_id is not None
        parameter_infos = self._params.get_parameter_infos_by_index(indices=model_parameter.param_id)[0]
        assert parameter_infos.constraint_value == model_parameter.constraint_value, (
            parameter_infos.constraint_value,
            model_parameter.constraint_value,
        )
        assert parameter_infos.constraint_sigma == model_parameter.constraint_sigma, (
            parameter_infos.constraint_sigma,
            model_parameter.constraint_sigma,
        )

        self._params.add_constraint_to_parameter(
            param_id=model_parameter.param_id, constraint_value=value, constraint_sigma=sigma
        )

    def _validate_add_data_arguments(
        self, channels: Dict[str, DataInputType], channel_weights: Optional[Dict[str, WeightsInputType]]
    ) -> None:

        if not len(channels) == self.number_of_channels:
            raise ValueError(
                f"You provided data for {len(channels)} channels, "
                f"but the model has {self.number_of_channels} channels defined!"
            )
        if not all(ch.name in channels for ch in self._channels):
            raise ValueError(
                "The provided data channels do not match the ones defined in the model:"
                "Defined channels: \n\t-"
                + "\n\t-".join([c.name for c in self._channels])
                + "\nProvided channels: \n\t-"
                + "\n\t-".join([c for c in channels])
            )
        if channel_weights is not None:
            logging.warning(
                "You are adding weights to your data! This should only be done for evaluation on MC "
                "(e.g. for Asimov fits or toy studies) so be sure to check if this is the case!"
            )
            if not list(channel_weights.keys()) == list(channels.keys()):
                raise ValueError(
                    f"The keys of the dictionaries provided for the parameter 'channels' and "
                    f"'channel_weights' do not match!\nKeys of channels dictionary: {channels.keys()}\n"
                    f"Keys of channel_weights dictionary: {channel_weights.keys()}"
                )

    def add_data(
        self,
        channels: Dict[str, DataInputType],
        channel_weights: Optional[Dict[str, WeightsInputType]] = None,
    ) -> None:
        self._check_is_not_finalized()
        assert self._data_channels.is_empty
        if self._has_data is True:
            raise RuntimeError(
                "Data has already been added to this model!\nThe following channels are registered:\n\t-"
                + "\n\t-".join(self._data_channels.data_channel_names)
            )

        self._validate_add_data_arguments(channels, channel_weights)

        for channel_name, channel_data in channels.items():
            mc_channel = self._channels.get_channel_by_name(name=channel_name)
            self._data_channels.add_channel(
                channel_name=channel_name,
                channel_data=channel_data,
                from_data=True,
                binning=mc_channel.binning,
                column_names=mc_channel.data_column_names,
                channel_weights=None if channel_weights is None else channel_weights[channel_name],
            )
        self._has_data = True

    def _create_single_template_component_from_template(self, template_input: Union[str, Template]) -> Component:
        if isinstance(template_input, Template):
            template = template_input
            source_info_text = "directly as template"
            assert template in self._template_manager.values(), (
                template.name,
                template.serial_number,
                self._template_manager,
            )
        elif isinstance(template_input, str):
            assert not template_input.startswith("temp_"), template_input
            if template_input.isdigit():
                template_identifier = int(template_input)  # type: Union[str, int]  # identifier is serial_number
                source_info_text = f"via the template serial number as 'temp_{template_input}'"
            else:
                template_identifier = template_input  # identifier is template name
                source_info_text = f"via the template name {template_input} as 'temp_{template_input}'"
            template = self.get_template(name_or_index=template_identifier)
        else:
            raise ValueError(
                f"The parameter 'template_input' must be either of type Template or str, "
                f"but you provided an object of type {type(template_input)}"
            )

        assert not any(template in comp.sub_templates for comp in self._component_manager.values()), "\n".join(
            [
                f"{comp.name}, {comp.name}: {[t.name for t in comp.sub_templates]}"
                for comp in self._component_manager.values()
                if template in comp.sub_templates
            ]
        )
        new_component_name = f"component_from_template_{template.name}"
        comp_serial_number_component_tuple = self.add_component(
            fraction_parameters=None,
            component=None,
            name=new_component_name,
            templates=[template],
            shared_yield=False,
        )

        assert isinstance(comp_serial_number_component_tuple, tuple), type(comp_serial_number_component_tuple).__name__
        comp_serial_number, component = comp_serial_number_component_tuple  # type: int, Component

        logging.info(
            f"New component with name {new_component_name} and serial_number {comp_serial_number} was created and "
            f"added directly from single template (serial_number: {template.serial_number}, name: {template.name}, "
            f"provided {source_info_text})."
        )

        return component

    def add_asimov_data_from_templates(
        self,
        round_bin_counts: bool = True,
    ) -> None:
        self._check_is_not_finalized()
        assert self._data_channels.is_empty
        if self._has_data:
            raise RuntimeError(
                "Data has already been added to this model!\nThe following channels are registered:\n\t-"
                + "\n\t-".join(self._data_channels.data_channel_names)
            )

        for channel in self._channels:
            channel_data = None
            for template in channel.templates:
                if channel_data is None:
                    channel_data = copy.copy(template.bin_counts)
                else:
                    channel_data += template.bin_counts

            self._data_channels.add_channel(
                channel_name=channel.name,
                channel_data=np.ceil(channel_data) if round_bin_counts else channel_data,
                from_data=False,
                binning=channel.binning,
                column_names=channel.data_column_names,
                channel_weights=None,
            )
        self._has_data = True

    def add_toy_data_from_templates(
        self,
        round_bin_counts: bool = True,
    ) -> None:
        if self._original_data_channels is None and self._data_channels is not None:
            # Backing up original data_channels
            self._original_data_channels = self._data_channels

        self._data_channels = ModelDataChannels()

        for channel in self._channels:
            channel_data = None  # type: Optional[np.ndarray]
            for template in channel.templates:
                if channel_data is None:
                    channel_data = copy.copy(template.bin_counts)
                else:
                    assert channel_data.shape == template.bin_counts.shape, (
                        channel_data.shape,
                        template.bin_counts.shape,
                    )
                    channel_data += template.bin_counts

            if round_bin_counts:
                channel_data = np.ceil(channel_data)

            toy_data = scipy_stats.poisson.rvs(channel_data, random_state=self._random_state)

            self._data_channels.add_channel(
                channel_name=channel.name,
                channel_data=np.ceil(toy_data) if round_bin_counts else toy_data,
                from_data=False,
                binning=channel.binning,
                column_names=channel.data_column_names,
                channel_weights=None,
            )

        # Resetting data bin count and data bin error cache
        self._data_stat_errors_sq = None
        self._data_channels._data_bin_counts = None

        self._has_data = True

    # endregion

    # region Uncertainty handling
    @property
    def systematics_covariance_matrices_per_channel(self) -> List[np.ndarray]:
        assert self._systematics_covariance_matrices_per_channel is not None
        return self._systematics_covariance_matrices_per_channel

    @property
    def inverse_template_bin_correlation_matrix(self) -> np.ndarray:
        assert self._inverse_template_bin_correlation_matrix is not None
        return self._inverse_template_bin_correlation_matrix

    @immutable_cached_property
    def floating_nuisance_parameter_indices(self) -> List[int]:
        all_bin_nuisance_parameter_indices = self.bin_nuisance_parameter_indices
        floating_nuisance_parameter_indices = []
        for all_index in all_bin_nuisance_parameter_indices:
            if self._params.floating_parameter_mask[all_index]:
                param_id = sum(self._params.floating_parameter_mask[: all_index + 1]) - 1  # type: int
                floating_nuisance_parameter_indices.append(param_id)

        return floating_nuisance_parameter_indices

    @immutable_cached_property
    def bin_nuisance_parameter_indices(self) -> List[int]:
        bin_nuisance_param_indices = self._params.get_parameter_indices_for_type(
            parameter_type=ParameterHandler.bin_nuisance_parameter_type,
        )

        if not self._bin_nuisance_params_checked:
            self._check_bin_nuisance_parameters()

        return bin_nuisance_param_indices

    @immutable_cached_property
    def relative_shape_uncertainties(self) -> np.ndarray:
        cov_matrices_per_ch_and_temp = self.systematics_covariance_matrices_per_channel
        # TODO: Maybe add some more checks...
        assert cov_matrices_per_ch_and_temp is not None

        assert self.template_bin_counts is not None

        # TODO: The following combination is only valid for Option 1b!

        padded_shape_uncertainties_per_ch = [
            np.stack(
                pad_sequences(
                    [np.sqrt(np.diag(cov_for_temp)) for cov_for_temp in temps_in_ch],
                    padding="post",
                    max_len=self.max_number_of_bins_flattened,
                    value=0.0,
                    dtype="float64",
                )
            )
            for temps_in_ch in cov_matrices_per_ch_and_temp
        ]

        shape_uncertainties = np.stack(padded_shape_uncertainties_per_ch)

        assert self.template_bin_counts.shape == shape_uncertainties.shape, (
            self.template_bin_counts.shape,
            shape_uncertainties.shape,
        )

        relative_shape_uncertainties = np.divide(
            shape_uncertainties,
            self.template_bin_counts,
            out=np.zeros_like(shape_uncertainties),
            where=self.template_bin_counts != 0.0,
        )

        return relative_shape_uncertainties

    @immutable_cached_property
    def _nuisance_matrix_shape(self) -> Tuple[int, ...]:
        return self.number_of_channels, max(self.number_of_templates), self.max_number_of_bins_flattened

    @immutable_cached_property
    def template_bin_errors_sq_per_ch_and_temp(self) -> List[List[np.ndarray]]:

        b_errs2_per_ch_and_temp = [[tmp.bin_errors_sq.flatten() for tmp in ch.templates] for ch in self._channels]

        # Resetting statistical error for empty bins to small value != 0:
        for stat_bin_errors_per_ch in b_errs2_per_ch_and_temp:
            for stat_bin_errors_per_temp in stat_bin_errors_per_ch:
                stat_bin_errors_per_temp[stat_bin_errors_per_temp == 0.0] = 1e-14

        b_errs2_per_channel = [np.stack(temps_in_ch) for temps_in_ch in b_errs2_per_ch_and_temp]
        self._check_template_bin_errors_shapes(bin_errors_per_ch=b_errs2_per_channel)

        return b_errs2_per_ch_and_temp

    @immutable_cached_property
    def template_stat_error_sq_matrix_per_channel(self) -> List[np.ndarray]:

        bin_errors_sq_per_ch_and_temp = self.template_bin_errors_sq_per_ch_and_temp
        bin_errors_sq_per_ch = [np.stack(temps_in_ch) for temps_in_ch in bin_errors_sq_per_ch_and_temp]
        stat_err_sq_matrix_per_ch = [
            np.apply_along_axis(func1d=np.diag, axis=1, arr=stat_err_array) for stat_err_array in bin_errors_sq_per_ch
        ]

        # Checking shape of matrices:
        assert len(stat_err_sq_matrix_per_ch) == self.number_of_channels, (
            len(stat_err_sq_matrix_per_ch),
            self.number_of_channels,
        )
        assert all(len(matrix.shape) == 3 for matrix in stat_err_sq_matrix_per_ch), (
            [m.shape for m in stat_err_sq_matrix_per_ch],
            [len(m.shape) for m in stat_err_sq_matrix_per_ch],
        )
        assert all(m.shape[:-1] == a.shape for m, a in zip(stat_err_sq_matrix_per_ch, bin_errors_sq_per_ch)), [
            (m.shape, a.shape) for m, a in zip(stat_err_sq_matrix_per_ch, bin_errors_sq_per_ch)
        ]
        assert all(m.shape[1] == m.shape[2] for m in stat_err_sq_matrix_per_ch), [
            m.shape for m in stat_err_sq_matrix_per_ch
        ]

        return stat_err_sq_matrix_per_ch

    @immutable_cached_property
    def template_stat_error_sq_matrix_per_ch_and_temp(self) -> List[List[np.ndarray]]:

        bin_errors_sq_per_ch_and_temp = self.template_bin_errors_sq_per_ch_and_temp
        stat_err_sq_matrix_per_ch_and_temp = [
            [np.diag(temp_array) for temp_array in stat_err_arrays_per_temp]
            for stat_err_arrays_per_temp in bin_errors_sq_per_ch_and_temp
        ]

        # Checking shape of matrices:
        assert len(stat_err_sq_matrix_per_ch_and_temp) == self.number_of_channels, (
            len(stat_err_sq_matrix_per_ch_and_temp),
            self.number_of_channels,
        )
        assert all(
            len(temps_per_ch) == ch.total_number_of_templates
            for temps_per_ch, ch in zip(stat_err_sq_matrix_per_ch_and_temp, self._channels)
        ), [(len(a), c.total_number_of_templates) for a, c in zip(stat_err_sq_matrix_per_ch_and_temp, self._channels)]
        assert all(isinstance(m, np.ndarray) for temps in stat_err_sq_matrix_per_ch_and_temp for m in temps), [
            type(m) for temps in stat_err_sq_matrix_per_ch_and_temp for m in temps
        ]
        assert all(len(m.shape) == 2 for temps in stat_err_sq_matrix_per_ch_and_temp for m in temps), [
            (m.shape, len(m.shape)) for temps in stat_err_sq_matrix_per_ch_and_temp for m in temps
        ]
        assert all(
            all(m.shape[:-1] == a.shape for m, a in zip(ms, arrs))
            for ms, arrs in zip(stat_err_sq_matrix_per_ch_and_temp, bin_errors_sq_per_ch_and_temp)
        ), [
            (m.shape, a.shape)
            for ms, arrs in zip(stat_err_sq_matrix_per_ch_and_temp, bin_errors_sq_per_ch_and_temp)
            for m, a in zip(ms, arrs)
        ]
        assert all(m.shape[0] == m.shape[1] for temps in stat_err_sq_matrix_per_ch_and_temp for m in temps), [
            m.shape for temps in stat_err_sq_matrix_per_ch_and_temp for m in temps
        ]

        return stat_err_sq_matrix_per_ch_and_temp

    def _initialize_template_uncertainties(self) -> None:
        """
        This function has to initialize the matrices holding the systematic uncertainties as well as the correlation
        matrix used in the gauss_term to constrain the nuisance parameters.

        The form of these matrices depend on the way the systematics are handled.
        In general two kinds of systematics must be considered:
            1. the statistical uncertainties of the templates
            2. all other systematics defined by additional weights (up and down variations, multiple variations; see
               systematics class.).
        The first type is always present, the second has to be provided by the user via the systematics of
        the Template class.

        The systematics can be handled independent for each bin and channel/template (Option 1a and 1b)
            -> number of nuisance parameters =
                    (1a): n_bins * n_channels    or
                    (1b): n_bins * n_channels * n_templates_p_ch
        or independent from each other (Option 2)
            -> number of nuisance parameters = n_systematics
        or both (Option 3), with and without differentiation between templates:
            -> number of nuisance parameters =
                    (3a): n_bins * n_channels * n_systematics    or
                    (3b): n_bins * n_channels * n_templates_p_ch * n_systematics



        The resulting systematic uncertainty matrix is stored in self._systematic_uncertainty_matrix and
        added to
            - the templates in self.get_templates before they are normalized, or
            - the yields in self.calculate_expected_bin_count.
        The correlation matrix for the nuisance parameters is stored in self._inverse_template_bin_correlation_matrix.

        # TODO: Differentiate between relative errors and absolute errors

        # TODO: Differentiate between correlation matrix for the gauss term handling the nuisance parameter constraints
                and the covariance matrix used for the 'smearing' of the templates.

        # TODO: Complete documentation once fully implemented!

        :return: None
        """
        # TODO: Think about this: This can be done as block diagonal matrix
        #                           - with each block being a channel, and all systematics combined for each
        #                             channel in one matrix via distribution_utiliy.get_combined_covariance
        #                           - or with one block for each template via template.bin_correlation_matrix
        #                           - or with one block for each systematic via loop over template.systematics (would
        #                             require some new function, though, to get the cov/corr matrix for each sys...)

        # TODO: Must decide on a way the nuisance parameters are use! (see TODO note above!)
        #       The matrix shapes must be chosen accordingly!

        # Getting template_statistics_sys and sum over template axis to get right shape:
        # TODO: Whether the sum is needed depends on option 1a, 1b or 2)! -> Currently it is Option 1b!
        template_statistics_sys_per_ch = self.template_stat_error_sq_matrix_per_ch_and_temp
        # Option 1a:
        # template_statistics_sys_per_ch = [m.sum(axis=0) for m in self.template_stat_error_sq_matrix_per_channel

        # TODO: The following combination is only valid for Option 1b!
        other_sys_cov_matrix_per_ch_and_temp = [
            [
                temp.bin_covariance_matrix
                if temp.bin_nuisance_parameters is not None and temp.use_other_systematics
                else np.zeros_like(template_statistics_sys_per_ch[ch_i][temp_i])
                for temp_i, temp in enumerate(ch.sub_templates)
            ]
            for ch_i, ch in enumerate(self._channels)
        ]

        # Option 1a:
        # other_systematics_cov_matrix_per_ch = [
        #     np.sum([temp.bin_covariance_matrix if temp.bin_nuisance_parameters is not None
        #             else np.zeros_like(template_statistics_sys_per_ch[ch_i])
        #             for temp in ch.sub_templates], axis=0)
        #     for ch_i, ch in enumerate(self._channels)
        # ]

        assert all(
            all(stat_m.shape == o_cov_m.shape for stat_m, o_cov_m in zip(stat_ms, o_cov_ms))
            for stat_ms, o_cov_ms in zip(template_statistics_sys_per_ch, other_sys_cov_matrix_per_ch_and_temp)
        ), [
            [(a.shape, b.shape) for a, b in zip(stat_ms, o_cov_ms)]
            for stat_ms, o_cov_ms in zip(template_statistics_sys_per_ch, other_sys_cov_matrix_per_ch_and_temp)
        ]

        # Option 1a:
        # assert all(stat_m.shape == o_cov_m.shape
        #            for stat_m, o_cov_m in zip(template_statistics_sys_per_ch, other_systematics_cov_matrix_per_ch)), \
        #     [(a.shape, b.shape) for a, b in zip(template_statistics_sys_per_ch, other_systematics_cov_matrix_per_ch)]

        cov_matrices_per_ch = [
            [stat_sys + cov_matrix for stat_sys, cov_matrix in zip(stat_sys_covs, o_covs)]
            for stat_sys_covs, o_covs in zip(template_statistics_sys_per_ch, other_sys_cov_matrix_per_ch_and_temp)
        ]

        # Option 1a:
        # cov_matrices_per_ch = [
        #     stat_sys + cov_matrix
        #     for stat_sys, cov_matrix in zip(template_statistics_sys_per_ch, other_systematics_cov_matrix_per_ch)
        # ]

        self._systematics_covariance_matrices_per_channel = cov_matrices_per_ch

        # TODO: The following combination is only valid for Option 1b!
        inv_corr_matrices = [
            np.linalg.inv(cov2corr(cov_matrix))
            for cov_matrices_per_temp in cov_matrices_per_ch
            for cov_matrix in cov_matrices_per_temp
        ]
        self._inverse_template_bin_correlation_matrix = block_diag(*inv_corr_matrices)

        # Option 1a:
        # inv_corr_matrices = [np.linalg.inv(cov2corr(cov_matrix)) for cov_matrix in cov_matrices_per_ch]
        # self._inverse_template_bin_correlation_matrix = block_diag(*inv_corr_matrices)

    def _check_systematics_uncertainty_matrices(self) -> None:
        assert len(self.systematics_covariance_matrices_per_channel) == self.number_of_channels, (
            len(self.systematics_covariance_matrices_per_channel),
            self.number_of_channels,
        )

        # TODO: The checks must depend on option of handling systematics, currently only checks for Option 1b are done!
        assert all(
            len(sys_per_temp) == ch.total_number_of_templates
            for sys_per_temp, ch in zip(self.systematics_covariance_matrices_per_channel, self._channels)
        ), [
            (len(t), ch.total_number_of_templates)
            for t, ch in zip(self.systematics_covariance_matrices_per_channel, self._channels)
        ]
        assert all(
            all(isinstance(m, np.ndarray) for m in per_temp)
            for per_temp in self.systematics_covariance_matrices_per_channel
        ), [type(m) for ms in self.systematics_covariance_matrices_per_channel for m in ms]
        assert all(
            all(len(m.shape) == 2 for m in per_temp) for per_temp in self.systematics_covariance_matrices_per_channel
        ), [m.shape for ms in self.systematics_covariance_matrices_per_channel for m in ms]
        assert all(
            all(m.shape[0] == m.shape[1] for m in per_temp)
            for per_temp in self.systematics_covariance_matrices_per_channel
        ), [m.shape for ms in self.systematics_covariance_matrices_per_channel for m in ms]
        assert all(
            all(m.shape[0] == ch.binning.num_bins_total for m in per_temp)
            for per_temp, ch in zip(self.systematics_covariance_matrices_per_channel, self._channels)
        ), [
            (m.shape[0], ch.binning.num_bins_total)
            for ms, ch in zip(self.systematics_covariance_matrices_per_channel, self._channels)
            for m in ms
        ]

        # Option 1a:
        # assert all(len(m.shape) == 2 for m in self.systematics_covariance_matrices_per_channel), \
        #     [m.shape for m in self.systematics_covariance_matrices_per_channel]
        # assert all(m.shape[0] == m.shape[1] for m in self.systematics_covariance_matrices_per_channel), \
        #     [m.shape for m in self.systematics_covariance_matrices_per_channel]
        # assert all(m.shape[0] == ch.binning.num_bins_total
        #            for m, ch in zip(self.systematics_covariance_matrices_per_channel, self._channels)), \
        #     [(m.shape[0], ch.binning.num_bins_total)
        #      for m, ch in zip(self.systematics_covariance_matrices_per_channel, self._channels)]

    def _check_bin_correlation_matrix(self) -> None:
        inv_corr_mat = self.inverse_template_bin_correlation_matrix

        # TODO: The checks must depend on option of handling systematics, currently only checks for Option 1b are done!
        expected_dim = sum([ch.binning.num_bins_total * ch.total_number_of_templates for ch in self._channels])

        # Option 1a:
        # expected_dim = sum([ch.binning.num_bins_total for ch in self._channels])

        assert len(inv_corr_mat.shape) == 2, inv_corr_mat.shape
        assert inv_corr_mat.shape[0] == expected_dim, (inv_corr_mat.shape, expected_dim)
        assert inv_corr_mat.shape[0] == inv_corr_mat.shape[1], inv_corr_mat.shape

        # Checking if matrix is symmetric.
        assert np.allclose(inv_corr_mat, inv_corr_mat.T, rtol=1e-05, atol=1e-08), (inv_corr_mat, "Matrix not symmetric")

    def _check_template_bin_errors_shapes(self, bin_errors_per_ch: List[np.ndarray]) -> None:
        assert len(bin_errors_per_ch) == self.number_of_channels, (len(bin_errors_per_ch), self.number_of_channels)
        assert all(isinstance(a, np.ndarray) for a in bin_errors_per_ch), [type(a) for a in bin_errors_per_ch]
        assert all(len(a.shape) == 2 for a in bin_errors_per_ch), [a.shape for a in bin_errors_per_ch]
        assert all(a.shape[0] == ch.total_number_of_templates for a, ch in zip(bin_errors_per_ch, self._channels)), [
            (a.shape[0], ch.total_number_of_templates) for a, ch in zip(bin_errors_per_ch, self._channels)
        ]
        assert all(a.shape[1] == ch_b.num_bins_total for a, ch_b in zip(bin_errors_per_ch, self.binning)), [
            (a.shape[1], ch_b.num_bins_total) for a, ch_b in zip(bin_errors_per_ch, self.binning)
        ]
        assert all(not np.any(matrix == 0) for matrix in bin_errors_per_ch), [np.any(m == 0) for m in bin_errors_per_ch]

    def _create_bin_nuisance_parameters(
        self,
        template: Template,
    ) -> List[TemplateParameter]:
        bin_nuisance_model_params = []  # type: List[Union[ModelParameter, str]]
        bin_nuisance_model_param_indices = []  # type: List[int]

        # TODO now: Implement creation of nuisance parameters as required for option 1a!
        #           -> Think about how nuisance parameter must be handled in general to make all options possible!!!
        # TODO: General implementation of creation of nuisance parameters for different options of uncertainty handling!

        nuisance_sigma = 1.0  # type: float
        initial_nuisance_value = 0.0  # type: float

        for counter in range(template.num_bins_total):
            model_param_index, model_parameter = self.add_model_parameter(
                name=f"bin_nuisance_param_{counter}_for_temp_{template.name}",
                parameter_type=ParameterHandler.bin_nuisance_parameter_type,
                floating=True,
                initial_value=initial_nuisance_value,
                constrain_to_value=initial_nuisance_value,
                constraint_sigma=nuisance_sigma,
            )
            bin_nuisance_model_params.append(model_parameter)
            bin_nuisance_model_param_indices.append(model_param_index)

        bin_nuisance_template_paras = self._get_list_of_template_params(
            input_params=bin_nuisance_model_params,
            serial_numbers=template.serial_number,
            container_name=template.name,
            parameter_type=ParameterHandler.bin_nuisance_parameter_type,
            input_parameter_list_name="bin_nuisance_parameters",
        )

        logging.info(
            f"Created {len(bin_nuisance_model_params)} bin nuisance ModelParameters "
            f"for template '{template.name}' (serial no: {template.serial_number}) with "
            f"{template.num_bins_total} bins and shape {template.shape}.\n"
            f"Bin nuisance parameter indices = {bin_nuisance_model_param_indices}"
        )

        return bin_nuisance_template_paras

    # endregion

    # region Constraint-related methods and properties

    @property
    def constraint_indices(self) -> List[int]:
        return [c.constraint_index for c in self._constraint_container]

    @property
    def constraint_values(self) -> List[float]:
        return [c.central_value for c in self._constraint_container]

    @property
    def constraint_sigmas(self) -> List[float]:
        return [c.uncertainty for c in self._constraint_container]

    def _collect_parameter_constraints(self) -> None:
        indices, cvs, css = self._params.get_constraint_information()
        self._constraint_container.extend(Constraint(idx, cv, cs) for idx, cv, cs in zip(indices, cvs, css))

    # endregion

    # region Parameter handling
    @property
    def names_of_floating_parameters(self) -> Tuple[str, ...]:
        return self._params.get_floating_parameter_names()

    @property
    def types_of_floating_parameters(self) -> Tuple[str, ...]:
        return self._params.get_floating_parameter_types()

    def get_yield_parameter_names(self) -> Tuple[str, ...]:
        return self._params.get_yield_parameter_names()

    def get_parameter_index(
        self,
        parameter_name: str,
    ) -> int:
        return self._params.get_index(name=parameter_name)

    def get_model_parameter(self, name_or_index: Union[str, int]) -> ModelParameter:
        if isinstance(name_or_index, int):
            assert name_or_index < len(self._model_parameters), (name_or_index, len(self._model_parameters))
            return self._model_parameters[name_or_index]
        elif isinstance(name_or_index, str):
            assert name_or_index in self._model_parameters_mapping, (
                name_or_index,
                self._model_parameters_mapping.keys(),
            )
            return self._model_parameters[self._model_parameters_mapping[name_or_index]]
        else:
            raise ValueError(
                f"Expected string or integer for argument 'name_or_index'\n"
                f"However, {name_or_index} of type {type(name_or_index)} was provided!"
            )

    def get_yield_parameter_name_from_process(
        self,
        process_name: str,
    ) -> str:
        templates = self.get_templates_by_process_name(process_name=process_name)
        _first_yield_param = templates[0].yield_parameter  # type: Optional[TemplateParameter]
        p_name = _first_yield_param.name if _first_yield_param is not None else None  # type: Optional[str]
        p_id = _first_yield_param.param_id if _first_yield_param is not None else None  # type: Optional[int]
        assert p_name is not None
        assert all(t.yield_parameter is not None and t.yield_parameter.name == p_name for t in templates), (
            process_name,
            [t.yield_parameter.name if t.yield_parameter is not None else None for t in templates],
        )
        assert all(t.yield_parameter is not None and t.yield_parameter.param_id == p_id for t in templates), (
            process_name,
            [t.yield_parameter.param_id if t.yield_parameter is not None else None for t in templates],
        )
        return p_name

    def get_yield(
        self,
        process_name: str,
    ) -> float:
        parameter_name = self.get_yield_parameter_name_from_process(process_name=process_name)
        return self._params.get_parameters_by_name(parameter_names=parameter_name)

    def set_initial_parameter_value(
        self,
        parameter_name: str,
        new_initial_value: float,
    ) -> None:
        self._params.set_parameter_initial_value(parameter_name=parameter_name, new_initial_value=new_initial_value)

    def set_yield(self, process_name: str, new_initial_value: float) -> None:
        parameter_name = self.get_yield_parameter_name_from_process(process_name=process_name)
        self.set_initial_parameter_value(parameter_name=parameter_name, new_initial_value=new_initial_value)

    def reset_initial_parameter_value(
        self,
        parameter_name: str,
    ) -> None:
        self._params.reset_parameter_initial_value(parameter_name=parameter_name)

    def reset_parameters_to_initial_values(self) -> None:
        self._params.reset_parameters_to_initial_values()

    def _get_yield_parameter_indices(self) -> List[int]:
        channel_with_max, max_number_of_templates = self._channel_with_max_number_of_templates
        _channel_with_max = self._channels[channel_with_max]  # type: Optional[Channel]
        assert _channel_with_max is not None
        _yield_params = [
            t.yield_parameter for t in _channel_with_max.templates if t.yield_parameter is not None
        ]  # type: List[TemplateParameter]
        assert len(_yield_params) == len(_channel_with_max.templates), "Undefined yield parameters encountered!"
        _yield_param_ids = [yp.param_id for yp in _yield_params if yp.param_id is not None]  # type: List[int]
        assert len(_yield_param_ids) == len(_channel_with_max.templates), "Undefined yield parameter ids encountered!"
        return _yield_param_ids

    def _get_list_of_template_params(
        self,
        input_params: List[Union[ModelParameter, str]],
        serial_numbers: Union[Tuple[int, ...], List[int], int],
        container_name: str,
        parameter_type: str,
        input_parameter_list_name: str,
    ) -> List[TemplateParameter]:
        assert (
            parameter_type in ParameterHandler.parameter_types
        ), f"parameter_type must be one of {ParameterHandler.parameter_types}, you provided {parameter_type}!"

        if isinstance(serial_numbers, int):
            if not parameter_type == ParameterHandler.bin_nuisance_parameter_type:
                raise ValueError(
                    f"For model parameters of a different type than "
                    f"parameter_type '{ParameterHandler.bin_nuisance_parameter_type}', a list or tuple of"
                    f"serial numbers must be provided via the argument 'serial_numbers'!"
                )
            serial_numbers = [serial_numbers] * len(input_params)
        else:
            if len(serial_numbers) != len(input_params):
                raise ValueError(
                    f"Length of 'serial_numbers' (={len(serial_numbers)}) must be the same as length"
                    f"of 'input_params' (={len(input_params)})!"
                )

        template_params = []
        for i, (input_parameter, temp_serial_number) in enumerate(zip(input_params, serial_numbers)):
            if isinstance(input_parameter, str):
                model_parameter = self.get_model_parameter(name_or_index=input_parameter)
            elif isinstance(input_parameter, ModelParameter):
                self._check_model_parameter_registration(model_parameter=input_parameter)
                model_parameter = input_parameter
            else:
                raise ValueError(
                    f"Encountered unexpected type {type(input_parameter)} " f"in the provided {input_parameter_list_name}"
                )

            if model_parameter.parameter_type != parameter_type:
                raise RuntimeError(
                    f"The ModelParameters provided via {input_parameter_list_name} must be of "
                    f"parameter_type '{parameter_type}', however, the {i + 1}th ModelParameter you "
                    f"provided is of parameter_type '{model_parameter.parameter_type}'..."
                )

            template_param = TemplateParameter(
                name=f"{container_name}_{model_parameter.name}",
                parameter_handler=self._params,
                model_parameter=model_parameter,
            )

            template_params.append(template_param)
            model_parameter.used_by(
                template_parameter=template_param,
                template_serial_number=temp_serial_number,
            )

        return template_params

    def _check_model_parameter_registration(self, model_parameter: ModelParameter) -> None:
        if model_parameter.parameter_handler is not self._params:
            raise ValueError(
                f"The model parameter you are trying to register uses a different ParameterHandler "
                f"than the model you are trying to register it to!\n"
                f"\tModel's ParameterHandler: {self._params}\n"
                f"\tModelParameters's ParameterHandler: {model_parameter.parameter_handler}\n"
            )
        if model_parameter.name not in self._model_parameters_mapping:
            raise RuntimeError(
                f"The model parameter you provided is not registered to the model you are "
                f"trying to use it in.\nYour model parameter has the following properties:\n"
                f"{model_parameter.as_string()}\n"
            )

    def _check_yield_parameters(self, yield_parameter_indices: Optional[List[int]]) -> None:
        indices = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.yield_parameter_type)

        if yield_parameter_indices is None:
            yield_parameter_indices = self._get_yield_parameter_indices()

        # Compare number of yield parameters in parameter handler, with what is used in the model
        if self.fraction_conversion.needed:
            assert len(yield_parameter_indices) >= len(indices), (len(yield_parameter_indices), len(indices))
        else:
            assert len(yield_parameter_indices) == len(indices), (len(yield_parameter_indices), len(indices))

        # Check number of yield parameters
        if not len(indices) == self.number_of_expected_independent_yields:
            raise RuntimeError(
                f"Number of yield parameters does not agree with the number of expected independent "
                f"yields for the model:\n\tnumber of"
                f"expected independent yields = {self.number_of_expected_independent_yields}\n\t"
                f"number of yield parameters: {len(indices)}\n\tnumber"
                f" of independent templates per channel: {self.number_of_independent_templates}"
                f"\n Defined yield parameters:\n"
                + "\n\n".join(
                    [
                        p.as_string()
                        for p in self._model_parameters
                        if p.parameter_type == ParameterHandler.yield_parameter_type
                    ]
                )
                + "\n\nYield parameters registered in Parameter Handler:\n"
                + "\n\n".join([i.as_string() for i in self._params.get_parameter_infos_by_index(indices=indices)])
            )

        channel_with_max, max_number_of_templates = self._channel_with_max_number_of_templates
        assert len(yield_parameter_indices) == max_number_of_templates, (
            len(yield_parameter_indices),
            max_number_of_templates,
        )

        for ch_count, channel in enumerate(self._channels):
            assert channel.total_number_of_templates <= len(yield_parameter_indices), (
                channel.total_number_of_templates,
                len(yield_parameter_indices),
            )
            for yield_parameter_index, template in zip(
                yield_parameter_indices[: channel.total_number_of_templates], channel.templates
            ):
                yield_parameter_info = self._params.get_parameter_infos_by_index(indices=yield_parameter_index)[0]
                model_parameter = self._model_parameters[yield_parameter_info.model_index]
                template_serial_number = model_parameter.usage_serial_number_list[ch_count]
                assert template.serial_number == template_serial_number, (template.serial_number, template_serial_number)

        # Check order of yield parameters:
        yield_parameter_infos = self._params.get_parameter_infos_by_index(indices=indices)
        used_model_parameter_indices = []  # type: List[int]
        template_serial_numbers_list = []  # type: List[List[int]]
        channel_orders = []  # type: List[List[int]]
        for yield_param_info in yield_parameter_infos:
            model_parameter_index = yield_param_info.model_index
            assert model_parameter_index not in used_model_parameter_indices, (
                model_parameter_index,
                used_model_parameter_indices,
            )
            used_model_parameter_indices.append(model_parameter_index)
            model_parameter = self._model_parameters[model_parameter_index]
            template_serial_numbers = model_parameter.usage_serial_number_list
            template_serial_numbers_list.append(template_serial_numbers)
            if len(template_serial_numbers) == 0:
                raise RuntimeError(f"Yield ModelParameter {yield_param_info.name} is not used!")

            channel_order = [
                c.channel_index for t in template_serial_numbers for c in self._channels if t in c.template_serial_numbers
            ]

            channel_orders.append(channel_order)

            if len(template_serial_numbers) < self.number_of_channels:
                logging.warning(
                    f"Yield ModelParameter {model_parameter.name} is used for less templates "
                    f"{len(template_serial_numbers)} than channels defined in the "
                    f"model {self.number_of_channels}!"
                )

        # Check order of channels:
        min_ind = self.min_number_of_independent_yields
        if not all(
            list(dict.fromkeys(channel_orders[0]))[:min_ind] == list(dict.fromkeys(co))[:min_ind] for co in channel_orders
        ):
            raise RuntimeError(
                "Channel order differs for different yield model parameters.\n\tParameter Name: Channel Order\n\t"
                + "\n\t".join([f"{p.name}: co" for co, p in zip(channel_orders, yield_parameter_infos)])
            )

        self._yields_checked = True

    def _check_efficiency_parameters(self) -> None:
        eff_i = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.efficiency_parameter_type)

        # Check the number of efficiency parameters, should be the same as the number of templates.
        assert len(eff_i) == self.total_number_of_templates, (len(eff_i), self.total_number_of_templates)

        efficiency_parameter_infos = self._params.get_parameter_infos_by_index(indices=eff_i)
        efficiency_model_parameters = [self._model_parameters[epi.model_index] for epi in efficiency_parameter_infos]

        # Check that the efficiency parameters are only used for one template each:
        templates_list = []
        for eff_param in efficiency_model_parameters:
            assert len(eff_param.usage_serial_number_list) == 1, eff_param.as_string()
            templates_list.append(eff_param.usage_serial_number_list[0])
        assert len(set(templates_list)) == len(
            templates_list
        ), f"{len(set(templates_list))}, {len(templates_list)}\n\n {templates_list}"

        # Check order of efficiency parameters:
        template_serial_numbers = [t for ch in self._channels for t in ch.template_serial_numbers]
        assert templates_list == template_serial_numbers, (templates_list, template_serial_numbers)

        self._efficiencies_checked = True

    # endregion

    # region Templates and Components
    @property
    def _channel_with_max_number_of_templates(self) -> Tuple[int, int]:
        channel_with_max, max_number_of_templates = max(enumerate(self.number_of_templates), key=operator.itemgetter(1))
        return channel_with_max, max_number_of_templates

    def get_template(
        self,
        name_or_index: Union[str, int],
    ) -> Template:

        return self._template_manager[name_or_index]

    def get_templates_by_process_name(
        self,
        process_name: str,
    ) -> List[Template]:

        return self._template_manager.get_fit_objects_by_process_name(process_name)

    def get_component(
        self,
        name_or_index: Union[str, int],
    ) -> Component:

        return self._component_manager[name_or_index]

    # endregion

    # region General checks
    def _check_fraction_parameters(self) -> None:
        self._fractions_checked = self._fraction_manager.check_fraction_parameters(self._model_parameters)

    def _check_matrix_shapes(
        self,
        yield_params: np.ndarray,
        fraction_params: np.ndarray,
        efficiency_params: np.ndarray,
        templates: np.ndarray,
    ) -> None:
        if self._is_checked:
            return

        assert len(yield_params.shape) == 1, (len(yield_params.shape), yield_params.shape)
        assert len(efficiency_params.shape) == 2, (len(efficiency_params.shape), efficiency_params.shape)
        assert len(templates.shape) == 3, (len(templates.shape), templates.shape)

        assert yield_params.shape[0] == efficiency_params.shape[1], (yield_params.shape[0], efficiency_params.shape[1])

        assert efficiency_params.shape[0] == templates.shape[0], (efficiency_params.shape[0], templates.shape[0])
        assert efficiency_params.shape[1] == templates.shape[1], (efficiency_params.shape[1], templates.shape[1])

        if self.fraction_conversion.needed:
            assert self._fraction_manager.check_matrix_shapes(yield_params, fraction_params)

    def _check_bin_nuisance_parameters(self) -> None:
        # TODO now: adapt nuisance parameter check to option 1a!
        # TODO: adapt nuisance parameter check to different options!

        nu_is = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.bin_nuisance_parameter_type)
        number_bins_with_nuisance = sum(
            [
                ch.binning.num_bins_total
                for ch in self._channels
                for t in ch.sub_templates
                if t.bin_nuisance_parameters is not None
            ]
        )
        assert len(nu_is) == number_bins_with_nuisance, (len(nu_is), number_bins_with_nuisance)

        nuisance_model_parameter_infos = self._params.get_parameter_infos_by_index(indices=nu_is)
        assert all(param_info.name.startswith("bin_nuisance_param_") for param_info in nuisance_model_parameter_infos), (
            f"Parameter names of parameters registered under type {ParameterHandler.bin_nuisance_parameter_type}:\n"
            + "\n\t".join([pi.name for pi in nuisance_model_parameter_infos])
        )

        index_counter = 0
        for channel in self._channels:
            for component in channel.components:
                for template in component.sub_templates:
                    if template.bin_nuisance_parameters is None:
                        assert not any(
                            pi.name.endswith(f"_for_temp_{template.name}") for pi in nuisance_model_parameter_infos
                        ), (
                            template.name,
                            [
                                pi.name
                                for pi in nuisance_model_parameter_infos
                                if pi.name.endswith(f"_for_temp_{template.name}")
                            ],
                        )
                    else:
                        n_bins = template.num_bins_total
                        current_bin_nu_pars = nuisance_model_parameter_infos[index_counter : index_counter + n_bins]
                        assert all(pi.name.endswith(f"_for_temp_{template.name}") for pi in current_bin_nu_pars), (
                            template.name,
                            n_bins,
                            [pi.name for pi in current_bin_nu_pars],
                        )
                        assert all(
                            pi.param_id == template.bin_nuisance_parameters[i].param_id
                            for i, pi in enumerate(current_bin_nu_pars)
                        ), (
                            template.name,
                            [
                                (
                                    pi.param_id,
                                    template.bin_nuisance_parameters[i].param_id,
                                    template.bin_nuisance_parameters[i].name,
                                )
                                for i, pi in enumerate(current_bin_nu_pars)
                            ],
                        )
                        index_counter += n_bins

        self._bin_nuisance_params_checked = True

    def _check_model_setup(self) -> None:
        info_text = (
            f"For consistency,\n"
            f"\t- the order of the templates/processes has to be the same in each channel, and\n"
            f"\t- Composition of all components has to be the same in each channel.\n"
            f"{self._model_setup_as_string()}"
        )

        _first_channel = self._channels[0]
        assert isinstance(_first_channel, Channel), type(_first_channel).__name__
        if not all(ch.process_names == _first_channel.process_names for ch in self._channels):
            raise RuntimeError(
                "The order of processes per channel, which make up the model is inconsistent!\n" + info_text
            )

        if not all(ch.process_names_per_component == _first_channel.process_names_per_component for ch in self._channels):
            raise RuntimeError("The order of channel components, which make up the model is inconsistent!\n" + info_text)

    def _check_has_data(
        self,
        adding: str,
    ) -> None:
        if self._has_data is True:
            raise RuntimeError(
                f"Trying to add new {adding} after adding the data to the model!\n"
                f"All {adding}s have to be added before 'add_data' is called!"
            )

    def _check_is_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "The model is not finalized, yet!\nPlease use the 'finalize_model' method to finalize the model setup!"
            )

    def _check_is_not_finalized(self) -> None:
        if self._is_initialized:
            raise RuntimeError("The Model has already been finalized and cannot be altered anymore!")

    # endregion

    # region Functions updated by the minimizer (speed sensitive, uses parameter_vector ndarray)

    def get_templates(
        self,
        nuisance_parameters: Optional[np.ndarray],
    ) -> np.ndarray:
        if nuisance_parameters is None and self._template_shape is not None:
            return self._template_shape

        if nuisance_parameters is not None:
            # Apply shape uncertainties:
            templates_with_shape_uncertainties = self.template_bin_counts * (
                1.0 + nuisance_parameters * self.relative_shape_uncertainties
            )
        else:
            templates_with_shape_uncertainties = self.template_bin_counts

        # Normalization of template bin counts with shape uncertainties to obtain the template shapes:
        norm_denominator = templates_with_shape_uncertainties.sum(axis=2)[:, :, np.newaxis]
        if len(np.argwhere(norm_denominator == 0)) > 0:
            # Handling cases where empty templates would cause division by 0, resulting in a template shape with NaNs.
            for row, col, _ in np.argwhere(norm_denominator == 0):
                assert all(templates_with_shape_uncertainties[row, col, :] == 0), templates_with_shape_uncertainties[
                    row, col, :
                ]
                norm_denominator[row, col, 0] = 1.0

        templates_with_shape_uncertainties /= norm_denominator

        if nuisance_parameters is None:
            self._template_shape = templates_with_shape_uncertainties

        return templates_with_shape_uncertainties

    def get_nuisance_parameters(
        self,
        parameter_vector: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nuisance_parameter_vector = self._params.get_combined_parameters_by_index(
            parameter_vector=parameter_vector, indices=self.bin_nuisance_parameter_indices
        )

        new_nuisance_matrix_shape = self._nuisance_matrix_shape
        complex_reshaping_required = not all(
            n_bins == self.number_of_bins_flattened_per_channel[0] for n_bins in self.number_of_bins_flattened_per_channel
        )

        if complex_reshaping_required:
            num_bins_times_templates = np.array(self.number_of_bins_flattened_per_channel) * np.array(
                self.number_of_templates
            )

            cum_sum_of_bins = np.cumsum(num_bins_times_templates)
            split_indices, check_sum = np.split(cum_sum_of_bins, [len(cum_sum_of_bins) - 1])
            assert check_sum[0] == sum(num_bins_times_templates), (
                check_sum,
                sum(num_bins_times_templates),
                num_bins_times_templates,
            )

            assert len(nuisance_parameter_vector) == check_sum[0], (len(nuisance_parameter_vector), check_sum[0])

            nuisance_param_vectors_per_ch = np.split(nuisance_parameter_vector, split_indices)
            assert all(
                len(nus_v) == n_bins for nus_v, n_bins in zip(nuisance_param_vectors_per_ch, num_bins_times_templates)
            ), [(len(nv), nb) for nv, nb in zip(nuisance_param_vectors_per_ch, num_bins_times_templates)]

            padded_nuisance_param_vectors = pad_sequences(nuisance_param_vectors_per_ch, padding="post", dtype="int32")

            nuisance_parameter_matrix = np.reshape(padded_nuisance_param_vectors, newshape=new_nuisance_matrix_shape)
        else:
            nuisance_parameter_matrix = np.reshape(nuisance_parameter_vector, newshape=new_nuisance_matrix_shape)

        return nuisance_parameter_vector, nuisance_parameter_matrix

    def get_efficiencies_matrix(
        self,
        parameter_vector: np.ndarray,
    ) -> np.ndarray:
        # TODO: Should be normalized to 1 over all channels? Can not be done when set to be floating
        # TODO: Add constraint which ensures that they are normalized?
        # TODO: Would benefit from allowing constrains, e.g. let them float around MC expectation
        if self._efficiency_indices is not None:
            return self._get_shaped_efficiency_parameters(
                parameter_vector=parameter_vector,
                indices=self._efficiency_indices,
            )

        self._check_is_initialized()
        indices = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.efficiency_parameter_type)
        shaped_efficiency_matrix = self._get_shaped_efficiency_parameters(
            parameter_vector=parameter_vector,
            indices=indices,
        )

        if not self._efficiencies_checked:
            self._check_efficiency_parameters()

        self._efficiency_indices = indices
        return shaped_efficiency_matrix

    def _get_shaped_efficiency_parameters(self, parameter_vector: np.ndarray, indices: List[int]) -> np.ndarray:
        self._check_is_initialized()

        eff_params_array = self._params.get_combined_parameters_by_index(
            parameter_vector=parameter_vector,
            indices=indices,
        )

        if self._efficiency_reshaping_indices is None:
            ntpc = self.number_of_templates  # ntpc = Number of templates per channel
            self._efficiency_padding_required = not all(nt == ntpc[0] for nt in ntpc)
            self._efficiency_reshaping_indices = [
                sum([temps_in_ch for temps_in_ch in ntpc[: i + 1]]) for i in range(len(ntpc) - 1)
            ]

        eff_params_array_list = np.split(eff_params_array, self._efficiency_reshaping_indices)

        if self._efficiency_padding_required:
            shaped_effs_matrix = pad_sequences(eff_params_array_list, padding="post", dtype="float64")
        else:
            shaped_effs_matrix = np.stack(eff_params_array_list)

        if not self._efficiencies_checked:
            assert all(len(effs) == nt for effs, nt in zip(eff_params_array_list, self.number_of_templates)), (
                eff_params_array_list,
                [len(effs) for effs in eff_params_array_list],
                self.number_of_templates,
            )
            assert len(shaped_effs_matrix.shape) == 2, (len(shaped_effs_matrix.shape), shaped_effs_matrix.shape)
            assert shaped_effs_matrix.shape[0] == self.number_of_channels, (
                shaped_effs_matrix.shape,
                shaped_effs_matrix.shape[0],
                self.number_of_channels,
            )
            assert shaped_effs_matrix.shape[1] == max(self.number_of_templates), (
                shaped_effs_matrix.shape,
                shaped_effs_matrix.shape[1],
                max(self.number_of_templates),
            )

        return shaped_effs_matrix

    def get_fractions_vector(self, parameter_vector: np.ndarray) -> np.ndarray:
        if self._fraction_indices is not None:
            return self._params.get_combined_parameters_by_index(
                parameter_vector=parameter_vector,
                indices=self._fraction_indices,
            )

        self._check_is_initialized()
        indices = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.fraction_parameter_type)

        if not self._fractions_checked:
            self._check_fraction_parameters()

        self._fraction_indices = indices
        return self._params.get_combined_parameters_by_index(parameter_vector=parameter_vector, indices=indices)

    def get_yields_vector(
        self,
        parameter_vector: np.ndarray,
    ) -> np.ndarray:
        if self._yield_indices is not None:
            return self._params.get_combined_parameters_by_index(
                parameter_vector=parameter_vector,
                indices=self._yield_indices,
            )
        self._check_is_initialized()

        indices_from_temps = self._get_yield_parameter_indices()

        if not self._yields_checked:
            self._check_yield_parameters(yield_parameter_indices=indices_from_temps)

        self._yield_indices = indices_from_temps
        return self._params.get_combined_parameters_by_index(
            parameter_vector=parameter_vector,
            indices=indices_from_temps,
        )

    def calculate_expected_bin_count(
        self,
        parameter_vector: np.ndarray,
        nuisance_parameters: np.ndarray,
    ) -> np.ndarray:
        if not self._is_checked:
            assert isinstance(self.fraction_conversion, FractionConversionInfo), type(self.fraction_conversion)

        yield_parameters = self.get_yields_vector(parameter_vector=parameter_vector)
        fraction_parameters = self.get_fractions_vector(parameter_vector=parameter_vector)
        efficiency_parameters = self.get_efficiencies_matrix(parameter_vector=parameter_vector)
        normed_efficiency_parameters = efficiency_parameters  # TODO: Implement normalization of efficiencies!
        # TODO: Regarding normalization of efficiencies: If the efficiencies are normed,
        #       the efficiency loss due to the scope of the templates must be considered as well!
        #       Maybe this can be done in the templates...

        # TODO: add also rate uncertainties to yields!

        normed_templates = self.get_templates(nuisance_parameters=nuisance_parameters)

        self._check_matrix_shapes(
            yield_params=yield_parameters,
            fraction_params=fraction_parameters,
            efficiency_params=normed_efficiency_parameters,
            templates=normed_templates,
        )

        if self.fraction_conversion.needed:
            bin_count = np.einsum(
                "ij, ijk -> ik",
                (
                    yield_parameters
                    * (
                        self.fraction_conversion.conversion_matrix @ fraction_parameters
                        + self.fraction_conversion.conversion_vector
                    )
                    * normed_efficiency_parameters
                ),
                normed_templates,
            )
        else:
            bin_count = np.einsum("ij, ijk -> ik", (yield_parameters * normed_efficiency_parameters), normed_templates)

        if not self._is_checked:
            check_bin_count_shape(
                bin_count=bin_count,
                number_of_channels=self.number_of_channels,
                max_number_of_bins=self.max_number_of_bins_flattened,
                where="calculate_expected_bin_count",
            )
            self._is_checked = True

        return bin_count

    @jit(forceobj=True)
    def _gauss_term(
        self,
        bin_nuisance_parameter_vector: np.ndarray,
    ) -> float:
        if len(bin_nuisance_parameter_vector) == 0:
            return 0.0

        if not self._gauss_term_checked:
            assert len(bin_nuisance_parameter_vector.shape) == 1, bin_nuisance_parameter_vector.shape
            assert (
                len(self.inverse_template_bin_correlation_matrix.shape) == 2
            ), self.inverse_template_bin_correlation_matrix.shape
            assert len(bin_nuisance_parameter_vector) == self.inverse_template_bin_correlation_matrix.shape[0], (
                bin_nuisance_parameter_vector.shape,
                self.inverse_template_bin_correlation_matrix.shape,
            )
            assert len(bin_nuisance_parameter_vector) == self.inverse_template_bin_correlation_matrix.shape[1], (
                bin_nuisance_parameter_vector.shape,
                self.inverse_template_bin_correlation_matrix.shape,
            )
            self._gauss_term_checked = True

        return (
            bin_nuisance_parameter_vector @ self.inverse_template_bin_correlation_matrix @ bin_nuisance_parameter_vector
        )

    def _constraint_term(
        self,
        parameter_vector: np.ndarray,
    ) -> float:
        if not self._constraint_container:
            return 0.0

        constraint_pars = self._params.get_combined_parameters_by_index(
            parameter_vector=parameter_vector,
            indices=self.constraint_indices,
        )

        constraint_term = np.sum(((self.constraint_values - constraint_pars) / self.constraint_sigmas) ** 2)

        if not self._constraint_term_checked:
            assert isinstance(constraint_term, float), (constraint_term, type(constraint_term))
            self._constraint_term_checked = True

        return constraint_term

    def get_masked_data_bin_count(self, mc_bin_count: np.ndarray) -> np.ndarray:
        if not self._ignore_empty_mc_bins:
            return self._data_channels.get_flattened_data_bin_counts(
                number_of_channels=self.number_of_channels, max_number_of_model_bins=self.max_number_of_bins_flattened
            )

        # We assume that the yields are always > 0 and, hence, the masking will never change!
        if self._masked_data_bin_counts is None:
            data_bin_count = np.copy(
                self._data_channels.get_flattened_data_bin_counts(
                    number_of_channels=self.number_of_channels, max_number_of_model_bins=self.max_number_of_bins_flattened
                )
            )  # type: np.ndarray

            assert data_bin_count.shape == mc_bin_count.shape, (data_bin_count.shape, mc_bin_count.shape)
            data_bin_count[mc_bin_count == 0.0] = 0.0

            self._masked_data_bin_counts = data_bin_count

        return self._masked_data_bin_counts

    @jit(forceobj=True)
    def chi2(
        self,
        parameter_vector: np.ndarray,
        fix_nuisance_parameters: bool = False,
    ) -> float:
        if fix_nuisance_parameters:
            nuisance_parameter_vector, nuisance_parameter_matrix = (np.array([]), None)
        else:
            nuisance_parameter_vector, nuisance_parameter_matrix = self.get_nuisance_parameters(parameter_vector)

        expected_bin_count = self.calculate_expected_bin_count(
            parameter_vector=parameter_vector,
            nuisance_parameters=nuisance_parameter_matrix,
        )  # type: np.ndarray

        chi2_data_term = np.sum(
            (expected_bin_count - self.get_masked_data_bin_count(mc_bin_count=expected_bin_count)) ** 2
            / (2 * self._data_channels.get_squared_data_stat_errors()),
            axis=None,
        )

        if not self._chi2_calculation_checked:
            assert isinstance(chi2_data_term, float), (chi2_data_term, type(chi2_data_term))
            self._chi2_calculation_checked = True

        return (
            chi2_data_term
            + self._gauss_term(bin_nuisance_parameter_vector=nuisance_parameter_vector)
            + self._constraint_term(parameter_vector=parameter_vector)
        )

    def nll(
        self,
        parameter_vector: np.ndarray,
        fix_nuisance_parameters: bool = False,
    ) -> float:
        if fix_nuisance_parameters:
            nuisance_parameter_vector, nuisance_parameter_matrix = (np.array([]), None)
        else:
            nuisance_parameter_vector, nuisance_parameter_matrix = self.get_nuisance_parameters(parameter_vector)

        expected_bin_count = self.calculate_expected_bin_count(
            parameter_vector=parameter_vector,
            nuisance_parameters=nuisance_parameter_matrix,
        )

        data_bin_count = self.get_masked_data_bin_count(mc_bin_count=expected_bin_count)  # type: np.ndarray

        poisson_term = np.sum(
            expected_bin_count - data_bin_count - xlogyx(data_bin_count, expected_bin_count),
            axis=None,
        )

        if not self._nll_calculation_checked:
            assert isinstance(poisson_term, float), (poisson_term, type(poisson_term))
            self._nll_calculation_checked = True

        return poisson_term + 0.5 * (
            self._gauss_term(bin_nuisance_parameter_vector=nuisance_parameter_vector)
            + self._constraint_term(parameter_vector=parameter_vector)
        )

    def update_parameters(
        self,
        parameter_vector: np.ndarray,
    ) -> None:
        self._params.update_parameters(parameter_vector=parameter_vector)

    # endregion

    # region Cost function factories

    # Using AbstractCostFunction class name as type hint, before AbstractCostFunction is defined.
    def create_nll(
        self,
        fix_nuisance_parameters: bool = False,
    ) -> "AbstractCostFunction":
        return NLLCostFunction(self, parameter_handler=self._params, fix_nuisance_parameters=fix_nuisance_parameters)

    # Using AbstractCostFunction class name as type hint, before AbstractCostFunction is defined.
    def create_chi2(
        self,
        fix_nuisance_parameters: bool = False,
    ) -> "AbstractCostFunction":
        return Chi2CostFunction(self, parameter_handler=self._params, fix_nuisance_parameters=fix_nuisance_parameters)

    # endregion

    # region Plotting

    @property
    def mc_channels_to_plot(self) -> ModelChannels:
        if not self._is_initialized:
            raise RuntimeError("The FitModel is not fully initialized, yet!")
        return self._channels

    @property
    def data_channels_to_plot(self) -> ModelDataChannels:
        if not self._is_initialized:
            raise RuntimeError("The FitModel is not fully initialized, yet!")
        return self._data_channels

    # endregion

    # region Misc

    def finalize_model(self) -> None:
        if not self._has_data:
            raise RuntimeError(
                "You have not added data, yet, so the model can not be finalized, yet!\n"
                "Please use the 'add_data' method to add data for all defined channels:\n\t"
                + "\n\t".join(f"{i}. {c.name}" for i, c in enumerate(self._channels))
            )

        self._check_model_setup()
        self._fraction_manager.convert_fractions()
        self._collect_parameter_constraints()

        self._initialize_template_uncertainties()
        self._check_systematics_uncertainty_matrices()
        self._check_bin_correlation_matrix()

        self._check_yield_parameters(yield_parameter_indices=None)
        self._check_fraction_parameters()
        self._check_efficiency_parameters()

        self._params.finalize()

        self._is_initialized = True

        logging.info(self._model_setup_as_string())

    def _model_setup_as_string(self) -> str:
        output_string = ""
        for channel in self._channels:
            output_string += f"Channel {channel.channel_index}: '{channel.name}'\n"
            for component in channel:
                output_string += f"\tComponent {component.component_index}: '{component.name}'\n"
                for template in component.sub_templates:
                    output_string += (
                        f"\t\tTemplate {template.template_index}: '{template.name}' "
                        f"(Process: '{template.process_name})'\n"
                    )

        return output_string

    def __repr__(self) -> str:
        return self._model_setup_as_string()

    # endregion


class AbstractCostFunction(ABC):
    """
    Abstract base class for all cost function to estimate yields using the template method.
    """

    def __init__(
        self,
        model: FitModel,
        parameter_handler: ParameterHandler,
        fix_nuisance_parameters: bool = False,
    ) -> None:
        self._model = model  # type: FitModel
        self._params = parameter_handler  # type: ParameterHandler
        self._fix_nui_params = fix_nuisance_parameters  # type: bool

    @property
    def x0(self) -> np.ndarray:
        """ Returns initial parameters of the model """
        return self._params.get_initial_values_of_floating_parameters()

    @property
    def param_names(self) -> Tuple[str, ...]:
        return self._model.names_of_floating_parameters

    @property
    def param_types(self) -> Tuple[str, ...]:
        return self._model.types_of_floating_parameters

    @abstractmethod
    def __call__(self, x: np.ndarray, *args) -> float:
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract base class.")


class Chi2CostFunction(AbstractCostFunction):
    def __call__(self, x: np.ndarray, *args) -> float:
        return self._model.chi2(
            parameter_vector=x,
            fix_nuisance_parameters=self._fix_nui_params,
        )


class NLLCostFunction(AbstractCostFunction):
    def __call__(self, x: np.ndarray, *args) -> float:
        return self._model.nll(
            parameter_vector=x,
            fix_nuisance_parameters=self._fix_nui_params,
        )
