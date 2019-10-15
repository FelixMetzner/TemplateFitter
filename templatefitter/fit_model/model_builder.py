"""
Class which defines the fit model by combining templates and handles the computation.
"""

import logging
import numpy as np
from numba import jit
from scipy.linalg import block_diag
from abc import ABC, abstractmethod

from typing import Optional, Union, List, Tuple, NamedTuple

from templatefitter.utility import xlogyx
from templatefitter.plotter import old_plotting

from templatefitter.fit_model.template import Template
from templatefitter.fit_model.component import Component
from templatefitter.binned_distributions.binning import Binning
from templatefitter.fit_model.channel import ChannelContainer, Channel
from templatefitter.fit_model.parameter_handler import ParameterHandler, ModelParameter, TemplateParameter

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ModelBuilder"]


class FractionConversionInfo(NamedTuple):
    needed: bool
    conversion_matrix: np.ndarray
    conversion_vector: np.ndarray


# TODO: Not yet considered are Yield Ratios:
#       We could add a ratio parameter instead of a yield parameter for components for which
#       the ratio of two yields should be fitted.
#       This would require an additional parameter type and an additional vector in the
#       calculation in calculate_bin_count:
#           yield_vector * ratio_vector
#       where ratio vector holds ones except for the row of the component which is related to another component
#       via the ratio. For this component the respective yield must stand at the place of its own yield value, e.g.:
#           ratio = yield_2 / yield_1 -> yield_2 = ratio * yield_1
#           => (yield_i, yield_1, yield_1, yield_j, ...)^T * (1, 1, ratio, 1, ...)^T


class ModelBuilder:
    def __init__(
            self,
            data,  # TODO: Type hint
            parameter_handler: ParameterHandler,
    ):
        self._data = data  # TODO: not used, yet!
        self._params = parameter_handler

        self._model_parameters = []
        self._model_parameters_mapping = {}

        # TODO: Maybe define also dedicated containers for templates, model_parameters and components, as for channels!
        self._templates = []
        self._templates_mapping = {}

        self._components = []
        self._components_mapping = {}

        self._channels = ChannelContainer()

        self._fraction_conversion = None

        self._is_checked = False

        # TODO:
        # self.yield_indices = []

        # self.subfraction_indices = []
        # self.num_fractions = 0

        # self.constrain_indices = []
        # self.constrain_value = np.array([])
        # self.constrain_sigma = np.array([])

        # self.x_obs = data.bin_counts.flatten()
        # self.x_obs_errors = data.bin_errors.flatten()

        # self._inv_corr = np.array([])
        # self.bin_par_slice = (0, 0)

        # self._dim = None
        # self.shape = ()

    # TODO: Check that every template of a model uses the same ParameterHandler instance!
    # TODO: Possible Check: For first call of expected_events_per_bin: Check if template indices are ordered correctly.

    def add_model_parameter(
            self,
            name: str,
            parameter_type: str,
            floating: bool,
            initial_value: float
    ) -> Tuple[int, ModelParameter]:
        if name in self._model_parameters_mapping.keys():
            raise RuntimeError(f"The model parameter with the name {name} already exists!\n"
                               f"It has the following properties:\n"
                               f"{self._model_parameters[self._model_parameters_mapping[name]].as_string()}")
        model_index = len(self._model_parameters)

        model_parameter = ModelParameter(
            name=name,
            parameter_handler=self._params,
            parameter_type=parameter_type,
            model_index=model_index,
            floating=floating,
            initial_value=initial_value
        )
        self._model_parameters.append(model_parameter)
        self._model_parameters_mapping.update({name: model_index})
        return model_index, model_parameter

    def add_template(
            self,
            template: Template,
            yield_parameter: Union[ModelParameter, str],
            bin_parameters  # TODO!
    ) -> int:
        if template.name in self._templates_mapping.keys():
            raise RuntimeError(f"The template with the name {template.name} is already registered!\n"
                               f"It has the index {self._templates_mapping[template.name]}\n")

        if isinstance(yield_parameter, str):
            yield_model_parameter = self.get_model_parameter(name_or_index=yield_parameter)
        elif isinstance(yield_parameter, ModelParameter):
            yield_model_parameter = yield_parameter
            # TODO: Check if model_parameter is registered, if not: call add_model_parameter!
        else:
            raise ValueError(f"Expected to receive object of type string or ModelParameter "
                             f"for argument yield_parameter, but you provided object of type {type(yield_parameter)}!")

        if not yield_model_parameter.parameter_type == ParameterHandler.yield_parameter_type:
            raise ValueError(f"The ModelParameter provided for the template yield must be of parameter_type 'yield', "
                             f"however, the ModelParameter you provided is of parameter_type "
                             f"'{yield_model_parameter.parameter_type}'")

        yield_param = TemplateParameter(
            name=f"{template.name}_{yield_model_parameter.name}",
            parameter_handler=self._params,
            parameter_type=yield_model_parameter.parameter_type,
            floating=yield_model_parameter.floating,
            initial_value=yield_model_parameter.initial_value,
            index=yield_model_parameter.index,
        )

        serial_number = len(self._templates)
        template.serial_number = serial_number

        yield_model_parameter.used_by(template_parameter=yield_param, template_serial_number=serial_number)

        template.initialize_parameters(
            yield_parameter=yield_param,
            bin_parameter_indices=  # TODO
        )

        self._templates.append(template)
        self._templates_mapping.update({template.name: serial_number})

        return serial_number

    def add_component(
            self,
            fraction_parameters: Optional[List[Union[ModelParameter, str]]] = None,
            component: Optional[Component] = None,
            name: Optional[str] = None,
            templates: Optional[List[Union[int, str, Template]]] = None,
            shared_yield: Optional[bool] = None,
    ) -> Union[int, Tuple[int, Component]]:
        creates_new_component = False
        component_input_error_text = "You can either add an already prepared component or create a new one.\n" \
                                     "For the first option, the argument 'component' must be used;\n" \
                                     "For the second option, please use the arguments 'templates', 'name' and " \
                                     "'shared_yield' as you would when creating a new Component object.\n" \
                                     "In the latter case, the templates can also be provided via their name or " \
                                     "serial_number es defined for this model."

        if (component is None and templates is None) or (component is not None and templates is not None):
            raise ValueError(component_input_error_text)
        elif component is not None:
            if not isinstance(component, Component):
                raise ValueError(f"The argument 'component' must be of type 'Component', "
                                 f"but you provided an object of type {type(component)}")
            if name is not None and name != component.name:
                raise ValueError(f"You set the argument 'name' to {name} despite the provided component "
                                 f"already having a different name, which is {component.name}")
            if shared_yield is not None and (shared_yield != component.shared_yield):
                raise ValueError(f"You set the argument 'shared_yield' to {shared_yield} despite the provided"
                                 f" component already having shared_yield set to {component.shared_yield}.")
        elif templates is not None:
            if not (isinstance(templates, list)
                    and all(isinstance(t, Template) or isinstance(t, int) or isinstance(t, str) for t in templates)):
                raise ValueError("The argument 'templates 'takes a list of Templates, integers or strings, but you "
                                 "provided " + f"an object of type {type(templates)}" if not isinstance(templates, list)
                                 else f"a list containing the types {[type(t) for t in templates]}")
            if shared_yield is None:
                raise ValueError("If you want to directly create and add component, you have to specify whether the "
                                 "templates of the component shall share their yields via the boolean parameter "
                                 "'shared_yields'!")
            if name is None:
                raise ValueError("If you want to directly create and add component, you have to specify its name "
                                 "via the argument 'name'!")
            template_list = []
            for template in templates:
                if isinstance(template, Template):
                    template_list.append(template)
                elif isinstance(template, int) or isinstance(template, str):
                    template_list.append(self.get_template(name_or_index=template))
                else:
                    raise ValueError(f"Parameter 'templates' must be a list of strings, integer or Templates "
                                     f"to allow for the creation of a new component.\n"
                                     f"You provided a list containing the types {[type(t) for t in templates]}.")

            creates_new_component = True
            component = Component(
                templates=template_list,
                params=self._params,
                name=name,
                shared_yield=shared_yield
            )
        else:
            raise ValueError(component_input_error_text)

        if component.name in self._components_mapping.keys():
            raise RuntimeError(f"A component with the name {component.name} is already registered!\n"
                               f"It has the index {self._components_mapping[component.name]}\n")

        if component.required_fraction_parameters == 0:
            if not (fraction_parameters is None or len(fraction_parameters) == 0):
                raise ValueError(f"The component requires no fraction parameters, "
                                 f"but you provided {len(fraction_parameters)}!")
            fraction_params = None
        else:
            if not all(isinstance(p, ModelParameter) or isinstance(p, str) for p in fraction_parameters):
                raise ValueError(f"Expected to receive list of string or ModelParameters for argument "
                                 f"fraction_parameters, but you provided a list containing objects of the types "
                                 f"{[type(p) for p in fraction_parameters]}!")
            fraction_params = []
            for fraction_parameter, temp_serial_number in zip(fraction_parameters, component.template_serial_numbers):
                if isinstance(fraction_parameter, str):
                    fraction_model_parameter = self.get_model_parameter(name_or_index=fraction_parameter)
                elif isinstance(fraction_parameter, ModelParameter):
                    fraction_model_parameter = fraction_parameter
                    # TODO: Check if model_parameter is registered, if not: call add_model_parameter!
                else:
                    raise ValueError(f"Encountered unexpected type {type(fraction_parameter)} "
                                     f"in the provided fraction_parameters")
                fraction_param = TemplateParameter(
                    name=f"{component.name}_{fraction_model_parameter.name}",
                    parameter_handler=self._params,
                    parameter_type=fraction_model_parameter.parameter_type,
                    floating=fraction_model_parameter.floating,
                    initial_value=fraction_model_parameter.initial_value,
                    index=fraction_model_parameter.index,
                )
                fraction_model_parameter.used_by(
                    template_parameter=fraction_param,
                    template_serial_number=temp_serial_number
                )
                fraction_params.append(fraction_param)

        serial_number = len(self._components)
        component.serial_number = serial_number

        component.initialize_parameters(fraction_parameters=fraction_params)

        self._components.append(component)
        self._components_mapping.update({component.name: serial_number})

        if creates_new_component:
            return serial_number, component
        else:
            return serial_number

    def add_channel(
            self,
            efficiency_parameters: List[Union[ModelParameter, str]],
            channel: Optional[Channel] = None,
            name: Optional[str] = None,
            components: Optional[List[Union[int, str, Component]]] = None,
            # TODO: If we want to add templates via serial_number or name directly, we have to be able to
            #       differentiate template names/serial_numbers from component names/serial_numbers
    ) -> Union[int, Tuple[int, Channel]]:
        creates_new_channel = False

        input_error_text = "A channel can either be added by providing\n" \
                           "\t- an already prepared channel via the argument 'channel'\nor\n" \
                           "\t- a list of components (directly, via their names or via their serial_numbers)\n" \
                           "\t  and a name for the channel, using the arguments 'components' and 'name', respectively."

        if channel is not None and components is not None:
            raise ValueError(input_error_text)
        elif channel is not None:
            if not isinstance(channel, Channel):
                raise ValueError(f"The argument 'channel' must be of type 'Channel', "
                                 f"but you provided an object of type {type(channel)}")
            if name is not None and name != channel.name:
                raise ValueError(f"You set the argument 'name' to {name} despite the provided channel "
                                 f"already having a different name, which is {channel.name}")
        elif components is not None:
            if not (isinstance(components, list)
                    and all(isinstance(c, Component) or isinstance(c, int) or isinstance(c, str) for c in components)):
                raise ValueError("The argument 'components 'takes a list of Components, integers or strings, but you "
                                 "provided "
                                 + f"an object of type {type(components)}" if not isinstance(components, list)
                                 else f"a list containing the types {[type(c) for c in components]}")
            if name is None:
                raise ValueError("When directly creating and adding a channel, you have to set the argument 'name'!")

            component_list = []
            for component in components:
                if isinstance(component, Component):
                    component_list.append(component)
                elif isinstance(component, int) or isinstance(component, str):
                    component_list.append(self.get_component(name_or_index=component))
                else:
                    raise ValueError(f"Unexpected type {type(component)} for element of provided list of components.")

            creates_new_channel = True
            channel = Channel(params=self._params, name=name, components=component_list)
        else:
            raise ValueError(input_error_text)

        # TODO: implement channel.required_efficiency_parameters
        if not len(efficiency_parameters) == channel.required_efficiency_parameters:
            raise ValueError(f"The channel requires {channel.required_efficiency_parameters} efficiency parameters, "
                             f"but you provided {len(efficiency_parameters)}!")
        efficiency_params = []
        # TODO: implement channel.template_serial_numbers
        for efficiency_parameter, temp_serial_number in zip(efficiency_parameters, channel.template_serial_numbers):
            if isinstance(efficiency_parameter, str):
                efficiency_model_parameter = self.get_model_parameter(name_or_index=efficiency_parameter)
            elif isinstance(efficiency_parameter, ModelParameter):
                efficiency_model_parameter = efficiency_parameter
                # TODO: Check if model_parameter is registered, if not: call add_model_parameter!
            else:
                raise ValueError(f"Encountered unexpected type {type(efficiency_parameter)} "
                                 f"in the provided efficiency_parameters")
            efficiency_param = TemplateParameter(
                name=f"{channel.name}_{efficiency_model_parameter.name}",
                parameter_handler=self._params,
                parameter_type=efficiency_model_parameter.parameter_type,
                floating=efficiency_model_parameter.floating,
                initial_value=efficiency_model_parameter.initial_value,
                index=efficiency_model_parameter.index,
            )
            efficiency_model_parameter.used_by(
                template_parameter=efficiency_param,
                template_serial_number=temp_serial_number
            )
            efficiency_params.append(efficiency_param)

        # TODO: implement channel.initialize_parameters
        channel.initialize_parameters(efficiency_parameters=efficiency_params)

        channel_serial_number = self._channels.add_channel(channel=channel)

        if creates_new_channel:
            return channel_serial_number, channel
        else:
            return channel_serial_number

    def setup_model(self, channels: ChannelContainer):
        if not all(c.params is self._params for c in channels):
            raise RuntimeError("The used ParameterHandler instances are not the same!")

        # TODO: Needs rework, as self._channels is now initialized as empty ChannelContainer
        if self._channels is not None:
            raise RuntimeError("Model already has channels defined!")

        self._channels = channels

        # TODO: Initialize parameters...
        #           - set indices,
        #           - identify floating parameters and fixed ones,
        #           - differentiate between different parameter types in ParameterHandler? -> would probably be better,

        self._initialize_fraction_conversion()

        # TODO: Complete Model setup

    def get_yields_vector(self):
        # TODO: Yields are provided by parameter handler...
        # TODO: number of yields = number of components
        # TODO: Vector or matrix? <-> Use additional dimension for channels?
        pass

    def get_fractions_vector(self):
        # TODO: Fractions are provided by parameter handler...
        # TODO: Number of fractions = number of templates - number of multi template components
        # TODO: Are NOT DIFFERENT for different channels
        pass

    def _initialize_fraction_conversion(self):
        # Fraction conversion matrix and vector should be equal in all channels.
        # The matrices and vectors are generated for each channel, tested for equality and then stored once.

        conversion_matrices = []
        conversion_vectors = []
        for channel in self._channels:
            matrices_for_channel = []
            vectors_for_channel = []
            for component in channel:
                n_sub = component.number_of_subcomponents
                if component.has_fractions:
                    matrix_part1 = np.diag(np.ones(n_sub - 1))
                    matrix_part2 = -1 * np.ones(n_sub - 1)
                    matrix = np.vstack([matrix_part1, matrix_part2])
                    matrices_for_channel.append(matrix)
                    vector = np.zeros((n_sub, 1))
                    vector[-1][0] = 1.
                    vectors_for_channel.append(vector)
                else:
                    matrices_for_channel.append(np.zeros((n_sub, n_sub)))
                    vectors_for_channel.append(np.ones((n_sub, 1)))

            conversion_matrices.append(block_diag(*matrices_for_channel))
            conversion_vectors.append(np.vstack(vectors_for_channel))

        assert all(m.shape[0] == v.shape[0] for m, v in zip(conversion_matrices, conversion_vectors))
        assert all(m.shape[0] == n_f for m, n_f in zip(conversion_matrices, self.number_of_fraction_parameters))
        assert all(np.array_equal(m, conversion_matrices[0]) for m in conversion_matrices)
        assert all(np.array_equal(v, conversion_vectors[0]) for v in conversion_vectors)

        self._fraction_conversion = FractionConversionInfo(
            needed=(not all(conversion_vectors[0] == 1)),
            conversion_matrix=conversion_matrices[0],
            conversion_vector=conversion_vectors[0]
        )

    def get_efficiency_vector(self):
        # TODO: Efficiencies are provided by parameter handler...
        # TODO: Number of efficiencies = number of templates
        # TODO: Are DIFFERENT for different channels
        # TODO: Might be fixed (extracted from simulation) or floating
        # TODO: Should be normalized to 1 over all channels? Can not be done when set to be floating
        # TODO: Would benefit from allowing constrains, e.g. let them float around MC expectation
        pass

    def get_template_bin_counts(self):
        # TODO: Get initial (not normed) shapes from templates
        pass

    def create_templates(self):
        # TODO: Are normed after application of corrections, but this should be done in calculate_bin_count!
        # TODO: Based on template bin counts
        pass

    def calculate_bin_count(self):
        if not self._is_checked:
            assert self._fraction_conversion is not None
            assert isinstance(self._fraction_conversion, FractionConversionInfo), type(self._fraction_conversion)
            # TODO: Check shapes!

        # TODO: Define remaining variables

        # TODO: Remember: fraction_parameters are None for last template in a component with shared yields
        #  and always None for components with no shared yields.

        if self._fraction_conversion.needed:
            bin_count = yield_parameters * (
                    self._fraction_conversion.conversion_matrix * fraction_parameters
                    + self._fraction_conversion.conversion_vector
            ) * normed_efficiency_parameters * normed_templates
        else:
            bin_count = yield_parameters * (
                    self._fraction_conversion.conversion_matrix * fraction_parameters
                    + self._fraction_conversion.conversion_vector
            ) * normed_efficiency_parameters * normed_templates

        if not self._is_checked:
            assert bin_count is not None
            # TODO: Check output shape!

            self._is_checked = True

        return bin_count

    @property
    def number_of_channels(self) -> int:
        return len(self._channels)

    @property
    def binning(self) -> Tuple[Binning, ...]:
        return tuple(channel.binning for channel in self._channels)

    @property
    def number_of_components(self) -> Tuple[int, ...]:
        return tuple(len(channel) for channel in self._channels)

    @property
    def number_of_templates(self) -> Tuple[int, ...]:
        return tuple(sum([comp.number_of_subcomponents for comp in ch.components]) for ch in self._channels)

    @property
    def number_of_independent_templates(self) -> Tuple[int, ...]:
        return tuple(
            sum([1 if comp.shared_yield else comp.number_of_subcomponents for comp in ch.components])
            for ch in self._channels
        )

    @property
    def number_of_fraction_parameters(self) -> Tuple[int, ...]:
        return tuple(sum([comp.required_fraction_parameters for comp in ch.components]) for ch in self._channels)

    def get_model_parameter(self, name_or_index: Union[str, int]) -> ModelParameter:
        if isinstance(name_or_index, int):
            assert name_or_index < len(self._model_parameters), (name_or_index, len(self._model_parameters))
            return self._model_parameters[name_or_index]
        elif isinstance(name_or_index, str):
            assert name_or_index in self._model_parameters_mapping.keys(), \
                (name_or_index, self._model_parameters_mapping.keys())
            return self._model_parameters[self._model_parameters_mapping[name_or_index]]
        else:
            raise ValueError(f"Expected string or integer for argument 'name_or_index'\n"
                             f"However, {name_or_index} of type {type(name_or_index)} was provided!")

    def get_template(self, name_or_index: Union[str, int]) -> Template:
        if isinstance(name_or_index, int):
            assert name_or_index < len(self._templates), (name_or_index, len(self._templates))
            return self._templates[name_or_index]
        elif isinstance(name_or_index, str):
            assert name_or_index in self._templates_mapping.keys(), \
                (name_or_index, self._templates_mapping.keys())
            return self._templates[self._templates_mapping[name_or_index]]
        else:
            raise ValueError(f"Expected string or integer for argument 'name_or_index'\n"
                             f"However, {name_or_index} of type {type(name_or_index)} was provided!")

    def get_component(self, name_or_index: Union[str, int]) -> Component:
        if isinstance(name_or_index, int):
            assert name_or_index < len(self._components), (name_or_index, len(self._components))
            return self._components[name_or_index]
        elif isinstance(name_or_index, str):
            assert name_or_index in self._components_mapping.keys(), \
                (name_or_index, self._components_mapping.keys())
            return self._components[self._components_mapping[name_or_index]]
        else:
            raise ValueError(f"Expected string or integer for argument 'name_or_index'\n"
                             f"However, {name_or_index} of type {type(name_or_index)} was provided!")

    # TODO: The following stuff is not adapted, yet...

    def template_matrix(self):
        """ Creates the fixed template stack """
        fractions_per_template = [template._flat_bin_counts for template in self.templates.values()]

        self.template_fractions = np.stack(fractions_per_template)
        self.shape = self.template_fractions.shape

    def relative_error_matrix(self):
        errors_per_template = [template.errors() for template
                               in self.templates.values()]

        self.template_errors = np.stack(errors_per_template)

    def initialise_bin_pars(self):
        """ Add parameters for the template """

        bin_pars = np.zeros((self.num_bins * len(self.templates), 1))
        bin_par_names = []
        for template in self.templates.values():
            bin_par_names += ["{}_binpar_{}".format(template.name, i) for i in range(0, self.num_bins)]
        bin_par_indices = self._params.add_parameters(bin_pars, bin_par_names)
        self.bin_par_slice = (bin_par_indices[0], bin_par_indices[-1] + 1)

    @jit
    def expected_events_per_bin(self, bin_pars: np.ndarray, yields: np.ndarray, sub_pars: np.ndarray) -> np.ndarray:
        sys_pars = self._params.get_parameters_by_index(self.sys_pars)
        # compute the up and down errors for single par variations
        up_corr = np.prod(1 + sys_pars * (sys_pars > 0) * self.up_errors, 0)
        down_corr = np.prod(1 + sys_pars * (sys_pars < 0) * self.down_errors, 0)
        corrections = (1 + self.template_errors * bin_pars) * (up_corr + down_corr)
        sub_fractions = np.matmul(self.converter_matrix, sub_pars) + self.converter_vector
        fractions = self.template_fractions * corrections
        norm_fractions = fractions / np.sum(fractions, 1)[:, np.newaxis]
        expected_per_bin = np.sum(norm_fractions * yields * sub_fractions, axis=0)
        return expected_per_bin
        # compute overall correction terms
        # get sub template fractions into the correct form with the converter and additive part
        # normalised expected corrected fractions
        # determine expected amount of events in each bin

    def fraction_converter(self) -> None:
        """
        Determines the matrices required to transform the sub-template parameters
        """
        arrays = []
        additive = []
        count = 0
        for template in self.packed_templates.values():
            if template._num_templates == 1:
                arrays.append(np.zeros((1, self.num_fractions)))
                additive.append(np.ones((1, 1)))
            else:
                n_fractions = template._num_templates - 1
                array = np.identity(n_fractions)
                array = np.vstack([array, np.full((1, n_fractions), -1.)])
                count += n_fractions
                array = np.pad(array, ((0, 0), (count - n_fractions, self.num_fractions - count)), mode='constant')
                arrays.append(array)
                additive.append(np.vstack([np.zeros((n_fractions, 1)), np.ones((1, 1))]))
        print(arrays)
        print(additive)
        self.converter_matrix = np.vstack(arrays)
        self.converter_vector = np.vstack(additive)

    def add_constraint(self, name: str, value: float, sigma: float) -> None:
        self.constrain_indices.append(self._params.get_index(name))
        self.constrain_value = np.append(self.constrain_value, value)
        self.constrain_sigma = np.append(self.constrain_sigma, sigma)

    def x_expected(self) -> np.ndarray:
        yields = self._params.get_parameters_by_index(self.yield_indices)
        fractions_per_template = np.array([template.fractions() for template in self.templates.values()])
        return yields @ fractions_per_template

    def bin_pars(self) -> np.ndarray:
        return np.concatenate([template.get_bin_pars() for template in self.templates.values()])

    def _create_block_diag_inv_corr_mat(self) -> None:
        inv_corr_mats = [template.inv_corr_mat() for template in self.templates.values()]
        self._inv_corr = block_diag(*inv_corr_mats)

    def _constrain_term(self) -> float:
        constrain_pars = self._params.get_parameters_by_index(self.constrain_indices)
        chi2constrain = np.sum(((self.constrain_value - constrain_pars) / self.constrain_sigma) ** 2)
        assert isinstance(chi2constrain, float), type(chi2constrain)  # TODO: Remove this assertion for speed-up!
        return chi2constrain

    @jit
    def _gauss_term(self, bin_pars: np.ndarray) -> float:
        return bin_pars @ self._inv_corr @ bin_pars

    @jit
    def chi2(self, pars: np.ndarray) -> float:
        self._params.set_parameters(pars)

        yields = self._params.get_parameters_by_index(self.yield_indices).reshape(self.num_templates, 1)
        sub_pars = self._params.get_parameters_by_index(self.subfraction_indices).reshape(self.num_fractions, 1)
        bin_pars = self._params.get_parameters_by_slice(self.bin_par_slice)

        chi2 = self.chi2_compute(bin_pars, yields, sub_pars)
        return chi2

    @jit
    def chi2_compute(self, bin_pars: np.ndarray, yields: np.ndarray, sub_pars: np.ndarray) -> float:
        chi2data = np.sum(
            (self.expected_events_per_bin(bin_pars.reshape(self.shape), yields, sub_pars) - self.x_obs) ** 2
            / (2 * self.x_obs_errors ** 2)
        )

        assert isinstance(chi2data, float), type(chi2data)  # TODO: Remove this assertion for speed-up!

        chi2 = chi2data + self._gauss_term(bin_pars)  # + self._constrain_term()  # TODO: Check this
        return chi2

    def nll(self, pars: np.ndarray) -> float:
        self._params.set_parameters(pars)

        exp_evts_per_bin = self.x_expected()
        poisson_term = np.sum(exp_evts_per_bin - self.x_obs - xlogyx(self.x_obs, exp_evts_per_bin))
        assert isinstance(poisson_term, float), type(poisson_term)  # TODO: Remove this assertion for speed-up!

        nll = poisson_term + (self._gauss_term() + self._constrain_term()) / 2.
        return nll

    @staticmethod
    def _get_projection(ax: str, bc: np.ndarray) -> np.ndarray:
        # TODO: Is the mapping for x and y defined the wrong way around?
        x_to_i = {
            "x": 1,
            "y": 0
        }

        # TODO: use method provided by BinnedDistribution!
        return np.sum(bc, axis=x_to_i[ax])

    # TODO: Use histogram for plotting!
    def plot_stacked_on(self, ax, plot_all=False, **kwargs):
        plot_info = old_plotting.PlottingInfo(
            templates=self.templates,
            params=self._params,
            yield_indices=self.yield_indices,
            dimension=self._dim,
            projection_fct=self._get_projection,
            data=self.data,
            has_data=self.has_data
        )
        return old_plotting.plot_stacked_on(plot_info=plot_info, ax=ax, plot_all=plot_all, **kwargs)

    # TODO: Problematic; At the moment some sort of forward declaration is necessary for type hint...
    def create_nll(self) -> "CostFunction":
        return CostFunction(self, parameter_handler=self._params)


# TODO: Maybe relocate cost functions into separate sub-package;
#  however: CostFunction depends on ModelBuilder and vice versa ...
class AbstractTemplateCostFunction(ABC):
    """
    Abstract base class for all cost function to estimate yields using the template method.
    """

    def __init__(self, model: ModelBuilder, parameter_handler: ParameterHandler) -> None:
        self._model = model
        self._params = parameter_handler

    def x0(self) -> np.ndarray:
        """ Returns initial parameters of the model """
        return self._params.get_parameters()

    def param_names(self) -> List[str]:
        return self._params.get_parameter_names()

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract base class.")


class CostFunction(AbstractTemplateCostFunction):
    def __init__(self, model: ModelBuilder, parameter_handler: ParameterHandler) -> None:
        super().__init__(model=model, parameter_handler=parameter_handler)

    def __call__(self, x) -> float:
        return self._model.chi2(x)
