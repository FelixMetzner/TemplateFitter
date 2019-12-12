"""
Class which defines the fit model by combining templates and handles the computation.
"""

import logging
import operator
import numpy as np

from numba import jit
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from keras.preprocessing.sequence import pad_sequences
from typing import Optional, Union, List, Tuple, Dict, NamedTuple

from templatefitter.utility import xlogyx

from templatefitter.fit_model.template import Template
from templatefitter.fit_model.component import Component
from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.binned_distribution import DataInputType
from templatefitter.fit_model.channel import ChannelContainer, Channel, DataChannelContainer
from templatefitter.fit_model.parameter_handler import ParameterHandler, ModelParameter, TemplateParameter

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["FitModel"]


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

# TODO: Define method to plot initial and fitted model vs data distribution
#       => use histogram for plotting!

# TODO: Maybe the FitModel could produce a Model object, which is a container that holds all
#       the necessary information and can be used to recreate the model.


class FractionConversionInfo(NamedTuple):
    needed: bool
    conversion_matrix: np.ndarray
    conversion_vector: np.ndarray


class FitModel:
    def __init__(
            self,
            parameter_handler: ParameterHandler,
    ):
        self._params = parameter_handler

        self._model_parameters = []
        self._model_parameters_mapping = {}

        self._templates = []
        self._templates_mapping = {}

        self._components = []
        self._components_mapping = {}

        self._channels = ChannelContainer()

        self._data_channels = DataChannelContainer()
        self._data_bin_counts = None
        self._data_bin_count_checked = False
        self._data_stat_errors_sq = None
        self._data_stat_errors_checked = False

        self._fraction_conversion = None
        self._inverse_template_bin_correlation_matrix = None

        self._has_data = False
        self._is_initialized = False
        self._is_checked = False

        self._yield_indices = None
        self._yields_checked = False

        self._fraction_indices = None
        self._fractions_checked = False

        self._efficiency_indices = None
        self._efficiency_reshaping_indices = None
        self._efficiency_padding_required = True
        self._efficiencies_checked = False

        self._bin_nuisance_param_indices = None
        self._bin_nuisance_params_checked = False

        self._constraint_indices = None
        self._constraint_values = None
        self._constraint_sigmas = None
        self._has_constrained_parameters = False

        self._template_bin_counts = None
        self._template_shapes_checked = False

        self._gauss_term_checked = False
        self._constraint_term_checked = False
        self._chi2_calculation_checked = False
        self._nll_calculation_checked = False

        self._floating_nuisance_parameter_indices = None

    def add_model_parameter(
            self,
            name: str,
            parameter_type: str,
            floating: bool,
            initial_value: float,
            constrain_to_value: Optional[float] = None,
            constraint_sigma: Optional[float] = None
    ) -> Tuple[int, ModelParameter]:
        self._check_is_not_finalized()
        self._check_has_data(adding="model parameter")

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
            initial_value=initial_value,
            constrain_to_value=constrain_to_value,
            constraint_sigma=constraint_sigma
        )
        self._model_parameters.append(model_parameter)
        self._model_parameters_mapping.update({name: model_index})
        return model_index, model_parameter

    def _check_model_parameter_registration(self, model_parameter: ModelParameter):
        if model_parameter.parameter_handler is not self._params:
            raise ValueError(f"The model parameter you are trying to register uses a different ParameterHandler "
                             f"than the model you are trying to register it to!\n"
                             f"\tModel's ParameterHandler: {self._params}\n"
                             f"\tModelParameters's ParameterHandler: {model_parameter.parameter_handler}\n"
                             )
        if model_parameter.name not in self._model_parameters_mapping.keys():
            raise RuntimeError(f"The model parameter you provided is not registered to the model you are "
                               f"trying to use it in.\nYour model parameter has the following properties:\n"
                               f"{model_parameter.as_string()}\n")

    def add_template(
            self,
            template: Template,
            yield_parameter: Union[ModelParameter, str],
            use_bin_nuisance_parameters: bool = True
    ) -> int:
        self._check_is_not_finalized()
        self._check_has_data(adding="template")

        if template.name in self._templates_mapping.keys():
            raise RuntimeError(f"The template with the name {template.name} is already registered!\n"
                               f"It has the index {self._templates_mapping[template.name]}\n")

        if isinstance(yield_parameter, str):
            yield_model_parameter = self.get_model_parameter(name_or_index=yield_parameter)
        elif isinstance(yield_parameter, ModelParameter):
            self._check_model_parameter_registration(model_parameter=yield_parameter)
            yield_model_parameter = yield_parameter
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

        if use_bin_nuisance_parameters:
            bin_nuisance_paras = self._create_bin_nuisance_parameters(template=template)
            assert len(bin_nuisance_paras) == template.num_bins_total, \
                (len(bin_nuisance_paras), template.num_bins_total)
        else:
            bin_nuisance_paras = None

        template.initialize_parameters(
            yield_parameter=yield_param,
            bin_nuisance_parameters=bin_nuisance_paras
        )

        self._templates.append(template)
        self._templates_mapping.update({template.name: serial_number})

        return serial_number

    def _create_bin_nuisance_parameters(self, template: Template) -> List[TemplateParameter]:
        bin_nuisance_model_params = []
        bin_nuisance_model_param_indices = []

        initial_nuisance_value = 0.
        nuisance_sigma = 1.0

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
            input_parameter_list_name="bin_nuisance_parameters"
        )

        logging.info(f"Created {len(bin_nuisance_model_params)} bin nuisance ModelParameters "
                     f"for template '{template.name}' (serial no: {template.serial_number}) with "
                     f"{template.num_bins_total} bins and shape {template.shape}.\n"
                     f"Bin nuisance parameter indices = {bin_nuisance_model_param_indices}")

        return bin_nuisance_template_paras

    def add_component(
            self,
            fraction_parameters: Optional[List[Union[ModelParameter, str]]] = None,
            component: Optional[Component] = None,
            name: Optional[str] = None,
            templates: Optional[List[Union[int, str, Template]]] = None,
            shared_yield: Optional[bool] = None,
    ) -> Union[int, Tuple[int, Component]]:
        self._check_is_not_finalized()
        creates_new_component = False
        component_input_error_text = "You can either add an already prepared component or create a new one.\n" \
                                     "For the first option, the argument 'component' must be used;\n" \
                                     "For the second option, please use the arguments 'templates', 'name' and " \
                                     "'shared_yield' as you would when creating a new Component object.\n" \
                                     "In the latter case, the templates can also be provided via their name or " \
                                     "serial_number es defined for this model."

        self._check_has_data(adding="component")

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
            if shared_yield is None and len(templates) > 1:
                raise ValueError("If you want to directly create and add a component, you have to specify whether the "
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

            fraction_params = self._get_list_of_template_params(
                input_params=fraction_parameters,
                serial_numbers=component.template_serial_numbers,
                container_name=component.name,
                parameter_type=ParameterHandler.fraction_parameter_type,
                input_parameter_list_name="fraction_parameters"
            )

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
    ) -> Union[int, Tuple[int, Channel]]:
        self._check_is_not_finalized()
        creates_new_channel = False
        input_error_text = "A channel can either be added by providing\n" \
                           "\t- an already prepared channel via the argument 'channel'\nor\n" \
                           "\t- a list of components (directly, via their names or via their serial_numbers)\n" \
                           "\t    (or alternatively Templates or template serial_numbers or names with a prefix " \
                           "\t    'temp_', from which a single template component will be generated)" \
                           "\t  and a name for the channel, using the arguments 'components' and 'name', respectively."

        self._check_has_data(adding="channel")

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
                    and all(isinstance(c, Component) or isinstance(c, int) or isinstance(c, str)
                            or isinstance(c, Template) for c in components)):
                raise ValueError("The argument 'components 'takes a list of Components, integers, strings or Templates,"
                                 " but you provided "
                                 + f"an object of type {type(components)}" if not isinstance(components, list)
                                 else f"a list containing the types {[type(c) for c in components]}")
            if name is None:
                raise ValueError("When directly creating and adding a channel, you have to set the argument 'name'!")

            component_list = []
            for component in components:
                if isinstance(component, Component):
                    component_list.append(component)
                if isinstance(component, Template):
                    new_component = self._create_single_template_component_from_template(template_input=component)
                    component_list.append(new_component)
                elif isinstance(component, int) or isinstance(component, str):
                    if isinstance(component, str) and component.startswith("temp_"):
                        # Handling case where a template identifier (string or index with prefix 'temp_') was provided:
                        temp_name_or_index_str = component[len("temp_"):]
                        new_component = self._create_single_template_component_from_template(
                            template_input=temp_name_or_index_str
                        )
                        component_list.append(new_component)
                    else:
                        component_list.append(self.get_component(name_or_index=component))
                else:
                    raise ValueError(f"Unexpected type {type(component)} for element of provided list of components.")

            creates_new_channel = True
            channel = Channel(params=self._params, name=name, components=component_list)
        else:
            raise ValueError(input_error_text)

        if not len(efficiency_parameters) == channel.required_efficiency_parameters:
            raise ValueError(f"The channel requires {channel.required_efficiency_parameters} efficiency parameters, "
                             f"but you provided {len(efficiency_parameters)}!")

        efficiency_params = self._get_list_of_template_params(
            input_params=efficiency_parameters,
            serial_numbers=channel.template_serial_numbers,
            container_name=channel.name,
            parameter_type=ParameterHandler.efficiency_parameter_type,
            input_parameter_list_name="efficiency_parameters"
        )

        channel.initialize_parameters(efficiency_parameters=efficiency_params)
        channel_serial_number = self._channels.add_channel(channel=channel)

        if creates_new_channel:
            return channel_serial_number, channel
        else:
            return channel_serial_number

    def _create_single_template_component_from_template(self, template_input: Union[str, Template]) -> Component:
        if isinstance(template_input, Template):
            template = template_input
            source_info_text = "directly as template"
            assert template in self._templates, \
                (template.name, template.serial_number, [(t.name, t.serial_number) for t in self._templates])
        elif isinstance(template_input, str):
            assert not template_input.startswith("temp_"), template_input
            if template_input.isdigit():
                template_identifier = int(template_input)  # identifier is template serial_number
                source_info_text = f"via the template serial number as 'temp_{template_input}'"
            else:
                template_identifier = template_input  # identifier is template name
                source_info_text = f"via the template name {template_input} as 'temp_{template_input}'"
            template = self.get_template(name_or_index=template_identifier)
        else:
            raise ValueError(f"The parameter 'template_input' must be either of type Template or str, "
                             f"but you provided an object of type {type(template_input)}")

        assert not any(template in comp.sub_templates for comp in self._components), "\n".join(
            [f"{comp.name}, {comp.name}: {[t.name for t in comp.sub_templates]}"
             for comp in self._components if template in comp.sub_templates]
        )
        new_component_name = f"component_from_template_{template.name}"
        assert new_component_name not in self._components_mapping.keys(), \
            (new_component_name, self._components_mapping.keys())
        comp_serial_number, component = self.add_component(
            fraction_parameters=None,
            component=None,
            name=new_component_name,
            templates=[template],
            shared_yield=False
        )

        logging.info(
            f"New component with name {new_component_name} and serial_number {comp_serial_number} was created and "
            f"added directly from single template (serial_number: {template.serial_number}, name: {template.name}, "
            f"provided {source_info_text})."
        )

        return component

    def add_constraint(self, name: str, value: float, sigma: float) -> None:
        self._check_is_not_finalized()

        if name not in self._model_parameters_mapping.keys():
            raise ValueError(f"A ModelParameter with the name '{name}' was not added, yet, "
                             f"and hus a constrained cannot be applied to it!")

        model_parameter = self._model_parameters[self._model_parameters_mapping[name]]
        if model_parameter.constraint_value is not None:
            raise RuntimeError(f"The ModelParameter '{name}' already is constrained with the settings"
                               f"\n\tconstraint_value = {model_parameter.constraint_value}"
                               f"\n\tconstraint_sigma = {model_parameter.constraint_sigma}\n"
                               f"and thus your constrained (cv = {value}, cs = {sigma}) cannot be applied!")

        index = model_parameter.index
        parameter_infos = self._params.get_parameter_infos_by_index(indices=index)[0]
        assert parameter_infos.constraint_value == model_parameter.constraint_value, \
            (parameter_infos.constraint_value, model_parameter.constraint_value)
        assert parameter_infos.constraint_sigma == model_parameter.constraint_sigma, \
            (parameter_infos.constraint_sigma, model_parameter.constraint_sigma)

        self._params.add_constraint_to_parameter(index=index, constraint_value=value, constraint_sigma=sigma)

    def add_data(
            self,
            channels: Dict[str, DataInputType],
            channel_weights: Optional[Dict[str, WeightsInputType]] = None
    ) -> None:
        self._check_is_not_finalized()
        assert self._data_channels.is_empty
        if self._has_data is True:
            raise RuntimeError("Data has already been added to this model!\nThe following channels are registered:\n\t-"
                               + "\n\t-".join(self._data_channels.data_channel_names))
        if not len(channels) == len(self._channels):
            raise ValueError(f"You provided data for {len(channels)} channels, "
                             f"but the model has {len(self._channels)} channels defined!")
        if not all(ch.name in channels.keys() for ch in self._channels):
            raise ValueError(f"The provided data channels do not match the ones defined in the model:"
                             f"Defined channels: \n\t-" + "\n\t-".join([c.name for c in self._channels])
                             + "\nProvided channels: \n\t-" + "\n\t-".join([c for c in channels.keys()])
                             )
        if channel_weights is not None:
            logging.warning("You are adding weights to your data! This should only be done for evaluation on MC "
                            "(e.g. for Asimov fits or toy studies) so be sure to check if this is the case!")
            if not list(channel_weights.keys()) == list(channels.keys()):
                raise ValueError(f"The keys of the dictionaries provided for the parameter 'channels' and "
                                 f"'channel_weights' do not match!\nKeys of channels dictionary: {channels.keys()}\n"
                                 f"Keys of channel_weights dictionary: {channel_weights.keys()}")

        for channel_name, channel_data in channels.items():
            mc_channel = self._channels.get_channel_by_name(name=channel_name)
            self._data_channels.add_channel(
                channel_name=channel_name,
                channel_data=channel_data,
                channel_weights=None if channel_weights is None else channel_weights[channel_name],
                binning=mc_channel.binning,
                column_names=mc_channel.data_column_names
            )
        self._has_data = True

    def finalize_model(self):
        if not self._has_data:
            raise RuntimeError("You have not added data, yet, so the model can not be finalized, yet!\n"
                               "Please use the 'add_data' method to add data for all defined channels:\n\t"
                               + "\n\t".join(f"{i}. {c.name}" for i, c in enumerate(self._channels)))

        self._check_model_setup()

        self._initialize_fraction_conversion()
        self._check_fraction_conversion()

        self._initialize_parameter_constraints()
        self._check_parameter_constraints()

        self._initialize_template_bin_uncertainties()
        self._check_template_bin_uncertainties()

        self._check_yield_parameters(yield_parameter_indices=None)
        self._check_fraction_parameters()
        self._check_efficiency_parameters()

        self._params.finalize()

        self._is_initialized = True

        logging.info(self._model_setup_as_string())

    def _check_model_setup(self) -> None:
        info_text = f"For consistency,\n" \
                    f"\t- the order of the templates/processes has to be the same in each channel, and\n" \
                    f"\t- Composition of all components has to be the same in each channel.\n" \
                    f"{self._model_setup_as_string()}"

        if not all(ch.process_names == self._channels[0].process_names for ch in self._channels):
            raise RuntimeError("The order of processes per channel, which make up the model is inconsistent!\n"
                               + info_text)

        if not all(ch.process_names_per_component == self._channels[0].process_names_per_component
                   for ch in self._channels):
            raise RuntimeError("The order of channel components, which make up the model is inconsistent!\n"
                               + info_text)

    def _model_setup_as_string(self) -> str:
        output_string = ""
        for channel in self._channels:
            output_string += f"Channel {channel.channel_index}: '{channel.name}\n"
            for component in channel:
                output_string += f"\tComponent {component.component_index}: '{component.name}\n"
                for template in component.sub_templates:
                    output_string += f"\t\tTemplate {template.template_index}: '{template.name} " \
                                     f"(Process: {template.process_name})\n"

        return output_string

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
        assert all(m.shape[0] == n_f for m, n_f in zip(conversion_matrices, self.number_of_independent_templates))
        assert all(np.array_equal(m, conversion_matrices[0]) for m in conversion_matrices)
        assert all(np.array_equal(v, conversion_vectors[0]) for v in conversion_vectors)

        self._fraction_conversion = FractionConversionInfo(
            needed=(not all(conversion_vectors[0] == 1)),
            conversion_matrix=conversion_matrices[0],
            conversion_vector=conversion_vectors[0]
        )

    def _check_fraction_conversion(self):
        if self._check_if_fractions_are_needed():
            assert self._fraction_conversion.needed is True
            assert np.any(self._fraction_conversion.conversion_vector != 1), self._fraction_conversion.conversion_vector
            assert np.any(self._fraction_conversion.conversion_matrix != 0), self._fraction_conversion.conversion_matrix
            assert len(self._fraction_conversion.conversion_vector.shape) == 1, \
                self._fraction_conversion.conversion_vector.shape
            assert self._fraction_conversion.conversion_vector.shape[0] == max(self.number_of_templates), \
                (self._fraction_conversion.conversion_vector.shape[0],
                 max(self.number_of_templates), self.number_of_templates)
            assert len(self._fraction_conversion.conversion_matrix.shape) == 2, \
                self._fraction_conversion.conversion_matrix.shape
            assert self._fraction_conversion.conversion_matrix.shape[0] == max(self.number_of_templates), \
                (self._fraction_conversion.conversion_matrix.shape[0],
                 max(self.number_of_templates), self.number_of_templates)
            assert self._fraction_conversion.conversion_matrix.shape[1] == len(self._channels[0].fractions_mask), \
                (self._fraction_conversion.conversion_matrix.shape[1], len(self._channels[0].fractions_mask),
                 self._channels[0].fractions_mask)
        else:
            logging.info("Fraction parameters are not used, as no templates of the same channel share "
                         "a common yield parameter.")
            assert self._fraction_conversion.needed is False
            assert np.all(self._fraction_conversion.conversion_vector == 1), self._fraction_conversion.conversion_vector
            assert np.all(self._fraction_conversion.conversion_matrix == 0), self._fraction_conversion.conversion_matrix

            assert all(sum(c.required_fraction_parameters) == 0 for c in self._channels), \
                "\n".join([f"{c.name}: {c.required_fraction_parameters}" for c in self._channels])
            yields_i = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.yield_parameter_type)
            assert all(len(yields_i) >= c.total_number_of_templates for c in self._channels), \
                f"{len(yields_i)}\n" + "\n".join([f"{c.name}: {c.total_number_of_templates}" for c in self._channels])

    def _initialize_parameter_constraints(self):
        indices, cvs, css = self._params.get_constraint_information()
        self._constraint_indices = indices
        self._constraint_values = cvs
        self._constraint_sigmas = css
        if len(indices) > 0:
            self._has_constrained_parameters = True

    def _check_parameter_constraints(self):
        if self._has_constrained_parameters:
            assert len(self._constraint_indices) > 0, (self._has_constrained_parameters, self._constraint_indices)
        else:
            assert len(self._constraint_indices) == 0, (self._has_constrained_parameters, self._constraint_indices)

        assert len(self._constraint_indices) == len(self._constraint_values), \
            (len(self._constraint_indices), len(self._constraint_values))
        assert len(self._constraint_indices) == len(self._constraint_sigmas), \
            (len(self._constraint_indices), len(self._constraint_sigmas))

        self._constraints_checked = True

    def _initialize_template_bin_uncertainties(self) -> None:
        # TODO: Think about this: This can be done as block diagonal matrix
        #                           - with each block being a channel, and all systematics combined for each
        #                             channel in one matrix via distribution_utiliy.get_combined_covariance
        #                           - or with one block for each template via template.bin_correlation_matrix
        #                           - or with one block for each systematic via loop over template.systematics (would
        #                             require some now function, though to get the cov/corr matrix for each sys...)

        inv_corr_mats = [
            np.linalg.inv(template.bin_correlation_matrix)
            for channel in self._channels for template in channel.sub_templates
            if template.bin_nuisance_parameters is not None
        ]
        if len(inv_corr_mats) == 0:
            self._inverse_template_bin_correlation_matrix = np.ndarray(shape=(0, 0))
        else:
            self._inverse_template_bin_correlation_matrix = block_diag(*inv_corr_mats)

    def _check_template_bin_uncertainties(self) -> None:
        assert self._inverse_template_bin_correlation_matrix is not None
        inv_corr_mat = self._inverse_template_bin_correlation_matrix

        expected_dim = sum([temp.num_bins_total for ch in self._channels for temp in ch.sub_templates
                            if temp.bin_nuisance_parameters is not None])
        assert inv_corr_mat.shape[0] == expected_dim, (inv_corr_mat.shape, expected_dim)

        if expected_dim == 0:
            return

        assert len(inv_corr_mat.shape) == 2, inv_corr_mat.shape
        assert inv_corr_mat.shape[0] == inv_corr_mat.shape[1], inv_corr_mat.shape

        # Checking matrix is symmetric.
        assert np.allclose(inv_corr_mat, inv_corr_mat.T, rtol=1e-05, atol=1e-08), (inv_corr_mat, "Matrix not symmetric")

    def get_yields_vector(self, parameter_vector: np.ndarray) -> np.ndarray:
        if self._yield_indices is not None:
            return self._params.get_combined_parameters_by_index(
                parameter_vector=parameter_vector,
                indices=self._yield_indices
            )
        self._check_is_initialized()

        indices_from_temps = self._get_yield_parameter_indices()

        if not self._yields_checked:
            self._check_yield_parameters(yield_parameter_indices=indices_from_temps)

        self._yield_indices = indices_from_temps
        return self._params.get_combined_parameters_by_index(
            parameter_vector=parameter_vector,
            indices=indices_from_temps
        )

    def _get_yield_parameter_indices(self) -> List[int]:
        channel_with_max, max_number_of_templates = self._get_channel_with_max_number_of_templates()
        return [t.yield_parameter.index for t in self._channels[channel_with_max].templates]

    def _check_yield_parameters(self, yield_parameter_indices: Optional[List[int]]) -> None:
        indices = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.yield_parameter_type)

        if yield_parameter_indices is None:
            yield_parameter_indices = self._get_yield_parameter_indices()

        # Compare number of yield parameters in parameter handler, with what is used in the model
        if self._fraction_conversion.needed:
            assert len(yield_parameter_indices) >= len(indices), (len(yield_parameter_indices), len(indices))
        else:
            assert len(yield_parameter_indices) == len(indices), (len(yield_parameter_indices), len(indices))

        # Check number of yield parameters
        if not len(indices) == self.number_of_expected_independent_yields:
            raise RuntimeError(f"Number of yield parameters does not agree with the number of expected independent "
                               f"yields for the model:\n\tnumber of"
                               f"expected independent yields = {self.number_of_expected_independent_yields}\n\t"
                               f"number of yield parameters: {len(indices)}\n\tnumber"
                               f" of independent templates per channel: {self.number_of_independent_templates}"
                               f"\n Defined yield parameters:\n"
                               + "\n\n".join([p.to_string() for p in self._model_parameters
                                              if p.parameter_type == ParameterHandler.yield_parameter_type])
                               + "\n\nYield parameters registered in Parameter Handler:\n"
                               + "\n\n".join([i.as_string()
                                              for i in self._params.get_parameter_infos_by_index(indices=indices)])
                               )

        channel_with_max, max_number_of_templates = self._get_channel_with_max_number_of_templates()
        assert len(yield_parameter_indices) == max_number_of_templates, \
            (len(yield_parameter_indices), max_number_of_templates)

        for ch_count, channel in enumerate(self._channels):
            assert channel.total_number_of_templates <= len(yield_parameter_indices), \
                (channel.total_number_of_templates, len(yield_parameter_indices))
            for yield_parameter_index, template in zip(yield_parameter_indices[:channel.total_number_of_templates],
                                                       channel.templates):
                yield_parameter_info = self._params.get_parameter_infos_by_index(indices=yield_parameter_index)[0]
                model_parameter = self._model_parameters[yield_parameter_info.model_index]
                template_serial_number = model_parameter.usage_serial_number_list[ch_count]
                assert template.serial_number == template_serial_number, \
                    (template.serial_number, template_serial_number)

        # Check order of yield parameters:
        yield_parameter_infos = self._params.get_parameter_infos_by_index(indices=indices)
        used_model_parameter_indices = []
        template_serial_numbers_list = []
        channel_orders = []
        for yield_param_info in yield_parameter_infos:
            model_parameter_index = yield_param_info.model_index
            assert model_parameter_index not in used_model_parameter_indices, \
                (model_parameter_index, used_model_parameter_indices)
            used_model_parameter_indices.append(model_parameter_index)
            model_parameter = self._model_parameters[model_parameter_index]
            template_serial_numbers = model_parameter.usage_serial_number_list
            template_serial_numbers_list.append(template_serial_numbers)
            if len(template_serial_numbers) == 0:
                raise RuntimeError(f"Yield ModelParameter {yield_param_info.name} is not used!")

            channel_order = [c.channel_index for t in template_serial_numbers for c in self._channels
                             if t in c.template_serial_numbers]

            channel_orders.append(channel_order)

            if len(template_serial_numbers) < len(self._channels):
                logging.warning(f"Yield ModelParameter {model_parameter.name} is used for less templates "
                                f"{len(template_serial_numbers)} than channels defined in the "
                                f"model {len(self._channels)}!")

        # Check order of channels:
        min_ind = self.min_number_of_independent_yields
        if not all(list(dict.fromkeys(channel_orders[0]))[:min_ind] == list(dict.fromkeys(co))[:min_ind]
                   for co in channel_orders):
            raise RuntimeError(
                "Channel order differs for different yield model parameters.\n\tParameter Name: Channel Order\n\t"
                + "\n\t".join([f"{p.name}: co" for co, p in zip(channel_orders, yield_parameter_infos)])
            )

        self._yields_checked = True

    def get_fractions_vector(self, parameter_vector: np.ndarray) -> np.ndarray:
        if self._fraction_indices is not None:
            return self._params.get_combined_parameters_by_index(
                parameter_vector=parameter_vector,
                indices=self._fraction_indices
            )

        self._check_is_initialized()
        indices = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.fraction_parameter_type)

        if not self._fractions_checked:
            self._check_fraction_parameters()

        self._fraction_indices = indices
        return self._params.get_combined_parameters_by_index(parameter_vector=parameter_vector, indices=indices)

    def _check_fraction_parameters(self) -> None:
        # Check number of fraction parameters
        assert self.number_of_dependent_templates == self.number_of_fraction_parameters, \
            (self.number_of_dependent_templates, self.number_of_fraction_parameters)
        frac_i = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.fraction_parameter_type)
        assert max(self.number_of_dependent_templates) == len(frac_i), f"Required fraction_parameters = " \
                                                                       f"{max(self.number_of_dependent_templates)}\n" \
                                                                       f"Registered fraction model parameters = " \
                                                                       f"{len(frac_i)}"

        # Check that fraction parameters are the same for each channel
        assert all(nf == self.number_of_fraction_parameters[0] for nf in self.number_of_fraction_parameters), \
            self.number_of_fraction_parameters

        fraction_parameter_infos = self._params.get_parameter_infos_by_index(indices=frac_i)
        fraction_model_parameters = [self._model_parameters[fpi.model_index] for fpi in fraction_parameter_infos]

        # Check order and consistency of fraction parameters
        for channel in self._channels:
            par_i = 0
            comps_and_temps = [(c, t) for c in channel.components for t in c.sub_templates]

            assert len(comps_and_temps) == channel.total_number_of_templates, \
                (len(comps_and_temps), channel.total_number_of_templates)
            assert len(channel.fractions_mask) == channel.total_number_of_templates, \
                (len(channel.fractions_mask), channel.total_number_of_templates)

            last_mask_value = False
            for counter, ((component, template), mask_value) in enumerate(zip(comps_and_temps, channel.fractions_mask)):
                if mask_value:
                    assert component.has_fractions
                    assert fraction_parameter_infos[par_i].parameter_type == ParameterHandler.fraction_parameter_type, \
                        (par_i, fraction_parameter_infos[par_i].as_string())
                    temp_serial_num = fraction_model_parameters[par_i].usage_serial_number_list[channel.channel_index]
                    assert temp_serial_num == template.serial_number, (temp_serial_num, template.serial_number)
                    temp_param = fraction_model_parameters[par_i].usage_template_parameter_list[channel.channel_index]
                    assert template.fraction_parameter == temp_param, \
                        (template.fraction_parameter.as_string(), temp_param.as_string())

                elif (not mask_value) and last_mask_value:
                    assert component.has_fractions
                    assert component.template_serial_numbers[-1] == template.serial_number, \
                        (component.template_serial_numbers[-1], template.serial_number)
                    assert template.fraction_parameter is None
                    par_i += 1
                    assert par_i <= len(fraction_parameter_infos), (par_i, len(fraction_parameter_infos))
                else:
                    assert (not mask_value) and (not last_mask_value), (mask_value, last_mask_value)
                    assert not component.has_fractions
                    assert template.fraction_parameter is None

                if counter == len(comps_and_temps) - 1:
                    assert par_i == len(fraction_parameter_infos), (counter, par_i, len(fraction_parameter_infos))

                last_mask_value = mask_value

        self._fractions_checked = True

    def get_efficiencies_matrix(self, parameter_vector: np.ndarray) -> np.ndarray:
        # TODO: Should be normalized to 1 over all channels? Can not be done when set to be floating
        # TODO: Would benefit from allowing constrains, e.g. let them float around MC expectation
        #       -> implement add_constrains method!
        # TODO: Add constraint which ensures that they are normalized?
        if self._efficiency_indices is not None:
            return self._get_shaped_efficiency_parameters(
                parameter_vector=parameter_vector,
                indices=self._efficiency_indices
            )

        self._check_is_initialized()
        indices = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.efficiency_parameter_type)
        shaped_efficiency_matrix = self._get_shaped_efficiency_parameters(
            parameter_vector=parameter_vector,
            indices=indices
        )

        if not self._efficiencies_checked:
            self._check_efficiency_parameters()

        self._efficiency_indices = indices
        return shaped_efficiency_matrix

    def _get_shaped_efficiency_parameters(self, parameter_vector: np.ndarray, indices: List[int]) -> np.ndarray:
        self._check_is_initialized()

        eff_params_array = self._params.get_combined_parameters_by_index(
            parameter_vector=parameter_vector,
            indices=indices
        )

        if self._efficiency_reshaping_indices is None:
            ntpc = self.number_of_templates  # ntpc = Number of templates per channel
            self._efficiency_padding_required = not all(nt == ntpc[0] for nt in ntpc)
            self._efficiency_reshaping_indices = [sum([temps_in_ch for temps_in_ch in ntpc[:i + 1]])
                                                  for i in range(len(ntpc) - 1)]

        eff_params_array_list = np.split(eff_params_array, self._efficiency_reshaping_indices)

        if self._efficiency_padding_required:
            shaped_effs_matrix = pad_sequences(eff_params_array_list, padding='post')
        else:
            shaped_effs_matrix = np.stack(eff_params_array_list)

        if not self._efficiencies_checked:
            assert all(len(effs) == nt for effs, nt in zip(eff_params_array_list, self.number_of_templates)), \
                (eff_params_array_list, [len(effs) for effs in eff_params_array_list], self.number_of_templates)
            assert len(shaped_effs_matrix.shape) == 2, (len(shaped_effs_matrix.shape), shaped_effs_matrix.shape)
            assert shaped_effs_matrix.shape[0] == self.number_of_channels, \
                (shaped_effs_matrix.shape, shaped_effs_matrix.shape[0], self.number_of_channels)
            assert shaped_effs_matrix.shape[1] == max(self.number_of_templates), \
                (shaped_effs_matrix.shape, shaped_effs_matrix.shape[1], max(self.number_of_templates))

        return shaped_effs_matrix

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
        assert len(set(templates_list)) == len(templates_list), \
            f"{len(set(templates_list))}, {len(templates_list)}\n\n {templates_list}"

        # Check order of efficiency parameters:
        template_serial_numbers = [t for ch in self._channels for t in ch.template_serial_numbers]
        assert templates_list == template_serial_numbers, (templates_list, template_serial_numbers)

        self._efficiencies_checked = True

    def get_bin_nuisance_vector(self, parameter_vector: np.ndarray) -> np.ndarray:
        return self._params.get_combined_parameters_by_index(
            parameter_vector=parameter_vector,
            indices=self.get_bin_nuisance_parameter_indices()
        )

    def get_bin_nuisance_parameter_indices(self) -> List[int]:
        if self._bin_nuisance_param_indices is not None:
            return self._bin_nuisance_param_indices

        bin_nuisance_param_indices = self._params.get_parameter_indices_for_type(
            parameter_type=ParameterHandler.bin_nuisance_parameter_type
        )

        if not self._bin_nuisance_params_checked:
            self._check_bin_nuisance_parameters()

        self._bin_nuisance_param_indices = bin_nuisance_param_indices
        return bin_nuisance_param_indices

    def _check_bin_nuisance_parameters(self) -> None:
        nu_is = self._params.get_parameter_indices_for_type(parameter_type=ParameterHandler.bin_nuisance_parameter_type)
        number_bins_with_nuisance = sum([t.num_bin_total for ch in self._channels
                                         for t in ch.sub_templates if t.bin_nuisance_parameters is not None])
        assert len(nu_is) == number_bins_with_nuisance, (len(nu_is), number_bins_with_nuisance)

        nuisance_model_parameter_infos = self._params.get_parameter_infos_by_index(indices=nu_is)
        assert all(param_info.name.startswith("bin_nuisance_param_")
                   for param_info in nuisance_model_parameter_infos), \
            (f"Parameter names of parameters registered under type {ParameterHandler.bin_nuisance_parameter_type}:\n"
             + "\n\t".join([pi.name for pi in nuisance_model_parameter_infos]))

        index_counter = 0
        for channel in self._channels:
            for component in channel.components:
                for template in component.sub_templates:
                    if template.bin_nuisance_parameters is None:
                        assert not any(
                            pi.name.endswith(f"_for_temp_{template.name}") for pi in nuisance_model_parameter_infos
                        ), (template.name, [pi.name for pi in nuisance_model_parameter_infos
                                            if pi.name.endswith(f"_for_temp_{template.name}")]
                            )
                    else:
                        n_bins = template.num_bins_total
                        current_bin_nu_pars = nuisance_model_parameter_infos[index_counter:index_counter + n_bins]
                        assert all(pi.name.endswith(f"_for_temp_{template.name}") for pi in current_bin_nu_pars), \
                            (template.name, n_bins, [pi.name for pi in current_bin_nu_pars])
                        assert all(pi.index == template.bin_nuisance_parameters[i].index
                                   for i, pi in enumerate(current_bin_nu_pars)), \
                            (template.name, [(pi.index, template.bin_nuisance_parameters[i].index,
                                              template.bin_nuisance_parameters[i].name)
                                             for i, pi in enumerate(current_bin_nu_pars)])
                        index_counter += n_bins

        self._bin_nuisance_params_checked = True

    def get_template_bin_counts(self):
        if self._template_bin_counts is not None:
            return self._template_bin_counts

        bin_counts_per_channel = [np.stack([tmp.bin_counts.flatten() for tmp in ch.templates]) for ch in self._channels]
        padded_bin_counts_per_channel = self._apply_padding_to_templates(bin_counts_per_channel=bin_counts_per_channel)
        template_bin_counts = np.stack(padded_bin_counts_per_channel)

        if not self._template_shapes_checked:
            self._check_template_shapes(template_bin_counts=template_bin_counts)

        self._template_bin_counts = template_bin_counts
        return self._template_bin_counts

    def _apply_padding_to_templates(self, bin_counts_per_channel: List[np.ndarray]) -> List[np.ndarray]:
        if not self._template_shapes_checked:
            assert all([bc.shape[1] == ch.binning.num_bins_total
                        for bc, ch in zip(bin_counts_per_channel, self._channels)]), "\t" + "\n\t".join(
                [f"{bc.shape[1]} : {ch.binning.num_bins_total}"
                 for bc, ch in zip(bin_counts_per_channel, self._channels)]
            )

        max_n_bins = max([bc.shape[1] for bc in bin_counts_per_channel])

        if all(bc.shape[1] == max_n_bins for bc in bin_counts_per_channel):
            return bin_counts_per_channel
        else:
            pad_widths = self._pad_widths_per_channel()
            return [
                np.pad(bc, pad_width=pad_width, mode='constant', constant_values=0)
                for bc, pad_width in zip(bin_counts_per_channel, pad_widths)
            ]

    def _pad_widths_per_channel(self) -> List[List[Tuple[int, int]]]:
        max_n_bins = max([ch.binning.num_bins_total for ch in self._channels])
        if not self._is_checked:
            assert max_n_bins == self.max_number_of_bins_flattened, (max_n_bins, self.max_number_of_bins_flattened)
        return [[(0, 0), (0, max_n_bins - ch.binning.num_bins_total)] for ch in self._channels]

    def _check_template_shapes(self, template_bin_counts: np.ndarray) -> None:
        # Check order of processes in channels:
        assert all(ch.process_names == self._channels[0].process_names for ch in self._channels), \
            [ch.process_names for ch in self._channels]

        # Check shape of template_bin_counts
        assert len(template_bin_counts.shape) == 3, (len(template_bin_counts.shape), template_bin_counts.shape)
        assert template_bin_counts.shape[0] == self.number_of_channels, \
            (template_bin_counts.shape, template_bin_counts.shape[0], self.number_of_channels)
        assert all(template_bin_counts.shape[1] == ts_in_ch for ts_in_ch in self.number_of_templates), \
            (template_bin_counts.shape, template_bin_counts.shape[1], [t_ch for t_ch in self.number_of_templates])
        assert template_bin_counts.shape[2] == self.max_number_of_bins_flattened, \
            (template_bin_counts.shape, template_bin_counts.shape[2], self.max_number_of_bins_flattened)

        self._template_shapes_checked = True

    def get_templates(self) -> np.ndarray:
        template_bin_counts = self.get_template_bin_counts()
        # TODO: Apply smearing based on bin_uncertainties.
        # TODO: Are normed after application of corrections, but this should be done in calculate_expected_bin_count!
        normed_smeared_templates = template_bin_counts

        # Temporary solution for normalization:
        normed_smeared_templates /= normed_smeared_templates.sum(axis=2)[:, :, np.newaxis]

        return normed_smeared_templates

    def calculate_expected_bin_count(self, parameter_vector: np.ndarray):
        if not self._is_checked:
            assert self._fraction_conversion is not None
            assert isinstance(self._fraction_conversion, FractionConversionInfo), type(self._fraction_conversion)

        yield_parameters = self.get_yields_vector(parameter_vector=parameter_vector)
        fraction_parameters = self.get_fractions_vector(parameter_vector=parameter_vector)
        efficiency_parameters = self.get_efficiencies_matrix(parameter_vector=parameter_vector)
        normed_efficiency_parameters = efficiency_parameters  # TODO: Implement normalization of efficiencies!

        normed_templates = self.get_templates()

        self._check_matrix_shapes(
            yield_params=yield_parameters,
            fraction_params=fraction_parameters,
            efficiency_params=normed_efficiency_parameters,
            templates=normed_templates
        )

        if self._fraction_conversion.needed:
            bin_count = np.einsum(
                "ij, ijk -> ik",
                (yield_parameters
                 * (self._fraction_conversion.conversion_matrix @ fraction_parameters
                    + self._fraction_conversion.conversion_vector)
                 * normed_efficiency_parameters),
                normed_templates
            )
        else:
            bin_count = np.einsum("ij, ijk -> ik", (yield_parameters * normed_efficiency_parameters), normed_templates)

        if not self._is_checked:
            self._check_bin_count_shape(bin_count=bin_count, where="calculate_expected_bin_count")
            self._is_checked = True

        return bin_count

    def _check_bin_count_shape(self, bin_count: np.ndarray, where: str) -> None:
        assert bin_count is not None, where
        assert len(bin_count.shape) == 2, (where, bin_count.shape, len(bin_count.shape))
        assert bin_count.shape[0] == self.number_of_channels, \
            (where, bin_count.shape, bin_count.shape[0], self.number_of_channels)
        assert bin_count.shape[1] == self.max_number_of_bins_flattened, \
            (where, bin_count.shape, bin_count.shape[1], self.max_number_of_bins_flattened)

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

        assert yield_params.shape[0] == efficiency_params.shape[1], \
            (yield_params.shape[0], efficiency_params.shape[1])

        assert efficiency_params.shape[0] == templates.shape[0], (efficiency_params.shape[0], templates.shape[0])
        assert efficiency_params.shape[1] == templates.shape[1], (efficiency_params.shape[1], templates.shape[1])

        if self._fraction_conversion.needed:
            assert len(fraction_params.shape) == 1, (len(fraction_params.shape), fraction_params.shape)
            assert len(self._fraction_conversion.conversion_vector.shape) == 1, \
                self._fraction_conversion.conversion_vector.shape
            assert len(self._fraction_conversion.conversion_matrix.shape) == 2, \
                self._fraction_conversion.conversion_matrix.shape
            assert yield_params.shape[0] == len(self._fraction_conversion.conversion_vector), \
                (yield_params.shape[0], len(self._fraction_conversion.conversion_vector))
            assert self._fraction_conversion.conversion_matrix.shape[0] == yield_params.shape[0], \
                (self._fraction_conversion.conversion_matrix.shape[0], yield_params.shape[0])
            assert self._fraction_conversion.conversion_matrix.shape[1] == fraction_params.shape[0], \
                (self._fraction_conversion.conversion_matrix.shape[1], fraction_params.shape[0])

    def get_flattened_data_bin_counts(self) -> np.array:
        if self._data_bin_counts is not None:
            return self._data_bin_counts

        flat_data_bin_counts = [data_channel.bin_counts.flatten() for data_channel in self._data_channels]
        padded_flat_data_bin_counts = self._apply_padding_to_data_bin_count(bin_counts_per_channel=flat_data_bin_counts)
        data_bin_count_matrix = np.stack(padded_flat_data_bin_counts)

        if self._data_channels.requires_rounding_due_to_weights:
            data_bin_count_matrix = np.ceil(data_bin_count_matrix)

        if not self._data_bin_count_checked:
            self._check_bin_count_shape(bin_count=data_bin_count_matrix, where="get_flattened_data_bin_counts")
            self._data_bin_count_checked = True

        self._data_bin_counts = data_bin_count_matrix
        return data_bin_count_matrix

    def get_squared_data_stat_errors(self) -> np.ndarray:
        if self._data_stat_errors_sq is not None:
            return self._data_stat_errors_sq

        flat_data_stat_errors_sq = [data_channel.bin_errors_sq.flatten() for data_channel in self._data_channels]
        padded_flat_data_stat_errors_sq = self._apply_padding_to_data_bin_count(
            bin_counts_per_channel=flat_data_stat_errors_sq
        )
        data_stat_errors_sq = np.stack(padded_flat_data_stat_errors_sq)

        if not self._data_stat_errors_checked:
            assert data_stat_errors_sq.shape == self.get_flattened_data_bin_counts().shape, \
                (data_stat_errors_sq.shape, self.get_flattened_data_bin_counts().shape)
            self._data_stat_errors_checked = True

        self._data_stat_errors_sq = data_stat_errors_sq
        return data_stat_errors_sq

    def _apply_padding_to_data_bin_count(self, bin_counts_per_channel: List[np.ndarray]) -> List[np.ndarray]:
        if not self._data_bin_count_checked:
            assert all(len(bc.shape) == 1 for bc in bin_counts_per_channel), [bc.shape for bc in bin_counts_per_channel]
            assert all(len(bc) == b.num_bins_total for bc, b in zip(bin_counts_per_channel, self.binning)), \
                "\n".join([f"{ch.name}: data_bins = {len(bc)} --- template_bins = {b.num_bins_total}"
                           for ch, bc, b in zip(self._channels, bin_counts_per_channel, self.binning)])

        max_n_bins = max([len(bc) for bc in bin_counts_per_channel])

        if not self._data_bin_count_checked:
            assert max_n_bins == self.max_number_of_bins_flattened, (max_n_bins, self.max_number_of_bins_flattened)

        if all(len(bc) == max_n_bins for bc in bin_counts_per_channel):
            return bin_counts_per_channel
        else:
            pad_widths = [(0, max_n_bins - ch.binning.num_bins_total) for ch in self._channels]
            return [
                np.pad(bc, pad_width=pad_width, mode='constant', constant_values=0)
                for bc, pad_width in zip(bin_counts_per_channel, pad_widths)
            ]

    @property
    def number_of_channels(self) -> int:
        return len(self._channels)

    @property
    def binning(self) -> Tuple[Binning, ...]:
        return tuple(channel.binning for channel in self._channels)

    @property
    def max_number_of_bins_flattened(self) -> int:
        return max(ch_binning.num_bins_total for ch_binning in self.binning)

    @property
    def number_of_components(self) -> Tuple[int, ...]:
        return tuple(len(channel) for channel in self._channels)

    @property
    def total_number_of_templates(self) -> int:
        return sum([c.total_number_of_templates for c in self._channels])

    @property
    # Number of templates per channel
    def number_of_templates(self) -> Tuple[int, ...]:
        return tuple(sum([comp.number_of_subcomponents for comp in ch.components]) for ch in self._channels)

    @property
    def number_of_independent_templates(self) -> Tuple[int, ...]:
        return tuple(
            sum([1 if comp.shared_yield else comp.number_of_subcomponents for comp in ch.components])
            for ch in self._channels
        )

    @property
    def number_of_dependent_templates(self) -> Tuple[int, ...]:
        return tuple([t - it for t, it in zip(self.number_of_templates, self.number_of_independent_templates)])

    @property
    def number_of_expected_independent_yields(self) -> int:
        return max(self.number_of_independent_templates)

    @property
    def min_number_of_independent_yields(self) -> int:
        return min(self.number_of_independent_templates)

    @property
    def number_of_fraction_parameters(self) -> Tuple[int, ...]:
        return tuple(sum([comp.required_fraction_parameters for comp in ch.components]) for ch in self._channels)

    def _get_channel_with_max_number_of_templates(self) -> Tuple[int, int]:
        channel_with_max, max_number_of_templates = max(enumerate(self.number_of_templates), key=operator.itemgetter(1))
        return channel_with_max, max_number_of_templates

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

    def _get_list_of_template_params(
            self,
            input_params: List[Union[ModelParameter, str]],
            serial_numbers: Union[Tuple[int, ...], List[int], int],
            container_name: str,
            parameter_type: str,
            input_parameter_list_name: str
    ):
        assert parameter_type in ParameterHandler.parameter_types, \
            f"parameter_type must be one of {ParameterHandler.parameter_types}, you provided {parameter_type}!"

        if isinstance(serial_numbers, int):
            if not parameter_type == ParameterHandler.bin_nuisance_parameter_type:
                raise ValueError(f"For model parameters of a different type than "
                                 f"parameter_type '{ParameterHandler.bin_nuisance_parameter_type}', a list or tuple of"
                                 f"serial numbers must be provided via the argument 'serial_numbers'!")
            serial_numbers = [serial_numbers] * len(input_params)
        else:
            if len(serial_numbers) != len(input_params):
                raise ValueError(f"Length of 'serial_numbers' (={len(serial_numbers)}) must be the same as length"
                                 f"of 'input_params' (={len(input_params)})!")

        template_params = []
        for i, (input_parameter, temp_serial_number) in enumerate(zip(input_params, serial_numbers)):
            if isinstance(input_parameter, str):
                model_parameter = self.get_model_parameter(name_or_index=input_parameter)
            elif isinstance(input_parameter, ModelParameter):
                self._check_model_parameter_registration(model_parameter=input_parameter)
                model_parameter = input_parameter
            else:
                raise ValueError(f"Encountered unexpected type {type(input_parameter)} "
                                 f"in the provided {input_parameter_list_name}")

            if model_parameter.parameter_type != parameter_type:
                raise RuntimeError(f"The ModelParameters provided via {input_parameter_list_name} must be of "
                                   f"parameter_type '{parameter_type}', however, the {i + 1}th ModelParameter you "
                                   f"provided is of parameter_type '{model_parameter.parameter_type}'...")

            template_param = TemplateParameter(
                name=f"{container_name}_{model_parameter.name}",
                parameter_handler=self._params,
                parameter_type=model_parameter.parameter_type,
                floating=model_parameter.floating,
                initial_value=model_parameter.initial_value,
                index=model_parameter.index,
            )

            template_params.append(template_param)
            model_parameter.used_by(
                template_parameter=template_param,
                template_serial_number=temp_serial_number
            )

        return template_params

    def _check_has_data(self, adding: str) -> None:
        if self._has_data is True:
            raise RuntimeError(f"Trying to add new {adding} after adding the data to the model!\n"
                               f"All {adding}s have to be added before 'add_data' is called!")

    def _check_is_initialized(self):
        if not self._is_initialized:
            raise RuntimeError(
                "The model is not finalized, yet!\nPlease use the 'finalize_model' method to finalize the model setup!"
            )

    def _check_is_not_finalized(self):
        if self._is_initialized:
            raise RuntimeError("The Model has already been finalized and cannot be altered anymore!")

    def _check_if_fractions_are_needed(self) -> bool:
        assert all(
            [n_it <= n_t for n_it, n_t in zip(self.number_of_independent_templates, self.number_of_templates)]
        ), "\n".join([f"{i}] <= {t}" for i, t in zip(self.number_of_independent_templates, self.number_of_templates)])
        return not (self.number_of_independent_templates == self.number_of_templates)

    @jit
    def _gauss_term(self, parameter_vector: np.ndarray) -> float:
        bin_nuisance_vector = self.get_bin_nuisance_vector(parameter_vector=parameter_vector)

        if len(bin_nuisance_vector) == 0:
            return 0.

        if not self._gauss_term_checked:
            assert len(bin_nuisance_vector.shape) == 1, bin_nuisance_vector.shape
            assert len(self._inverse_template_bin_correlation_matrix.shape) == 2, \
                self._inverse_template_bin_correlation_matrix.shape
            assert len(bin_nuisance_vector) == self._inverse_template_bin_correlation_matrix.shape[0], \
                (bin_nuisance_vector.shape, self._inverse_template_bin_correlation_matrix.shape)
            assert len(bin_nuisance_vector) == self._inverse_template_bin_correlation_matrix.shape[1], \
                (bin_nuisance_vector.shape, self._inverse_template_bin_correlation_matrix.shape)
            self._gauss_term_checked = True

        return bin_nuisance_vector @ self._inverse_template_bin_correlation_matrix @ bin_nuisance_vector

    def _constraint_term(self, parameter_vector: np.ndarray) -> float:
        if not self._has_constrained_parameters:
            return 0.

        constraint_pars = self._params.get_combined_parameters_by_index(
            parameter_vector=parameter_vector,
            indices=self._constraint_indices
        )

        constraint_term = np.sum(((self._constraint_values - constraint_pars) / self._constraint_sigmas) ** 2)

        if not self._constraint_term_checked:
            assert isinstance(constraint_term, float), (constraint_term, type(constraint_term))
            self._constraint_term_checked = True

        return constraint_term

    @jit
    def chi2(self, parameter_vector: np.ndarray) -> float:
        chi2_data_term = np.sum(
            (self.calculate_expected_bin_count(parameter_vector=parameter_vector)
             - self.get_flattened_data_bin_counts()) ** 2 / (2 * self.get_squared_data_stat_errors()),
            axis=None
        )

        if not self._chi2_calculation_checked:
            assert isinstance(chi2_data_term, float), (chi2_data_term, type(chi2_data_term))
            self._chi2_calculation_checked = True

        return (chi2_data_term + self._gauss_term(parameter_vector=parameter_vector)
                + self._constraint_term(parameter_vector=parameter_vector))

    def nll(self, parameter_vector: np.ndarray) -> float:
        expected_bin_count = self.calculate_expected_bin_count(parameter_vector=parameter_vector)
        poisson_term = np.sum(
            expected_bin_count - self.get_flattened_data_bin_counts()
            - xlogyx(self.get_flattened_data_bin_counts(), expected_bin_count),
            axis=None
        )

        if not self._nll_calculation_checked:
            assert isinstance(poisson_term, float), (poisson_term, type(poisson_term))
            self._nll_calculation_checked = True

        return poisson_term + 0.5 * (self._gauss_term(parameter_vector=parameter_vector)
                                     + self._constraint_term(parameter_vector=parameter_vector))

    # Using CostFunction class name as type hint, before CostFunction is defined.
    def create_nll(self) -> "CostFunction":
        return NLLCostFunction(self, parameter_handler=self._params)

    # Using CostFunction class name as type hint, before CostFunction is defined.
    def create_chi2(self) -> "CostFunction":
        return Chi2CostFunction(self, parameter_handler=self._params)

    def update_parameters(self, parameter_vector: np.ndarray) -> None:
        self._params.update_parameters(parameter_vector=parameter_vector)

    @property
    def floating_nuisance_parameter_indices(self) -> List[int]:
        if self._floating_nuisance_parameter_indices is not None:
            return self._floating_nuisance_parameter_indices
        all_bin_nuisance_parameter_indices = self.get_bin_nuisance_parameter_indices()
        floating_nuisance_parameter_indices = []
        for floating_index, all_index in enumerate(all_bin_nuisance_parameter_indices):
            if self._params.floating_parameter_mask[all_index]:
                floating_nuisance_parameter_indices.append(floating_index)

        self._floating_nuisance_parameter_indices = floating_nuisance_parameter_indices
        return floating_nuisance_parameter_indices

    @property
    def names_of_floating_parameters(self) -> List[str]:
        return self._params.get_floating_parameter_names()

    def set_initial_parameter_value(self, parameter_name: str, new_initial_value: float) -> None:
        self._params.set_parameter_initial_value(parameter_name=parameter_name, new_initial_value=new_initial_value)

    def reset_initial_parameter_value(self, parameter_name: str) -> None:
        self._params.reset_parameter_initial_value(parameter_name=parameter_name)

    @property
    def mc_channels_to_plot(self) -> ChannelContainer:
        if not self._is_initialized:
            raise RuntimeError("The FitModel is not fully initialized, yet!")
        return self._channels

    @property
    def data_channels_to_plot(self) -> DataChannelContainer:
        if not self._is_initialized:
            raise RuntimeError("The FitModel is not fully initialized, yet!")
        return self._data_channels

    # TODO: Remaining functions that have to be looked through if every old functionality is covered:
    # def relative_error_matrix(self):
    #     errors_per_template = [template.errors() for template
    #                            in self.templates.values()]
    #
    #     self.template_errors = np.stack(errors_per_template)
    #
    # @jit
    # def expected_events_per_bin(self, bin_pars: np.ndarray, yields: np.ndarray, sub_pars: np.ndarray) -> np.ndarray:
    #     sys_pars = self._params.get_parameters_by_index(self.sys_pars)
    #     # compute the up and down errors for single par variations
    #     up_corr = np.prod(1 + sys_pars * (sys_pars > 0) * self.up_errors, 0)
    #     down_corr = np.prod(1 + sys_pars * (sys_pars < 0) * self.down_errors, 0)
    #     corrections = (1 + self.template_errors * bin_pars) * (up_corr + down_corr)
    #     sub_fractions = np.matmul(self.converter_matrix, sub_pars) + self.converter_vector
    #     fractions = self.template_fractions * corrections
    #     norm_fractions = fractions / np.sum(fractions, 1)[:, np.newaxis]
    #     expected_per_bin = np.sum(norm_fractions * yields * sub_fractions, axis=0)
    #     return expected_per_bin
    #     # compute overall correction terms
    #     # get sub template fractions into the correct form with the converter and additive part
    #     # normalised expected corrected fractions
    #     # determine expected amount of events in each bin


class AbstractTemplateCostFunction(ABC):
    """
    Abstract base class for all cost function to estimate yields using the template method.
    """

    def __init__(self, model: FitModel, parameter_handler: ParameterHandler) -> None:
        self._model = model
        self._params = parameter_handler

    @property
    def x0(self) -> np.ndarray:
        """ Returns initial parameters of the model """
        return self._params.get_initial_values_of_floating_parameters()

    @property
    def param_names(self) -> List[str]:
        return self._model.names_of_floating_parameters

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract base class.")


class Chi2CostFunction(AbstractTemplateCostFunction):
    def __init__(self, model: FitModel, parameter_handler: ParameterHandler) -> None:
        super().__init__(model=model, parameter_handler=parameter_handler)

    def __call__(self, x) -> float:
        return self._model.chi2(x)


class NLLCostFunction(AbstractTemplateCostFunction):
    def __init__(self, model: FitModel, parameter_handler: ParameterHandler) -> None:
        super().__init__(model=model, parameter_handler=parameter_handler)

    def __call__(self, x) -> float:
        return self._model.nll(x)
