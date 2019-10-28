"""
The Component class is used to hold one or multiple templates which describe one component in a reconstruction channel.
If a component consists of multiple sub-components this class manages the respective fractions of the components.
Otherwise it acts just as a wrapper for the template class.
"""

import logging

from typing import Union, Optional, List, Tuple

from templatefitter.fit_model.template import Template
from templatefitter.binned_distributions.binning import Binning
from templatefitter.fit_model.parameter_handler import ParameterHandler, TemplateParameter

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Component", "ComponentTemplateInputType"]

ComponentTemplateInputType = Union[Template, List[Template]]


class Component:
    def __init__(
            self,
            templates: ComponentTemplateInputType,
            params: ParameterHandler,
            name: str,
            shared_yield: Optional[bool] = None,
    ):
        self._name = name
        self._templates = None
        self._binning = None
        self._params = params

        self._shared_yield = None
        self._has_fractions = None
        self._required_fraction_parameters = None

        self._initialize_templates(templates=templates)
        self._initialize_fractions(shared_yield=shared_yield)

        self._fraction_parameters = None

        self._component_serial_number = None

        self._component_index = None
        self._channel_index = None

    def _initialize_templates(self, templates: ComponentTemplateInputType) -> None:
        if isinstance(templates, Template):
            self._templates = (templates,)
            templates.template_index = 0
            self._binning = templates.binning
        elif isinstance(templates, list):
            if not all(isinstance(t, Template) for t in templates):
                raise ValueError("The parameter 'template' must be a Template or a List of Templates!\n"
                                 "You provided a list with the types:\n\t-"
                                 + "\n\t-".join([str(type(t)) for t in templates]))
            if not all(t.binning == templates[0].binning for t in templates):
                raise ValueError("All templates of a component must have the same binning.")
            if not all((t.data_column_names == templates[0].data_column_names if t.data_column_names is not None
                        else t.data_column_names is templates[0].data_column_names) for t in templates):
                raise RuntimeError("The data_column_names of the templates you are trying to combine in this component"
                                   " are not consistent:\n\t"
                                   + "\n\t".join([f"{t.name}: {t.data_column_names}" for t in templates]))
            self._templates = tuple(t for t in templates)
            for template_index, template in enumerate(templates):
                template.template_index = template_index
            self._binning = templates[0].binning
        else:
            raise ValueError(f"The parameter 'template' must be a Template or a List of Templates!\n"
                             f"You provided an object of type {type(templates)}.")

        if len(self._templates) < 1:
            raise RuntimeError(f"No templates have been added to the component! You provided templates = {templates}")
        if not all(t.params is self._params for t in self._templates):
            raise RuntimeError("The used ParameterHandler instances are not the same!")

    def _initialize_fractions(self, shared_yield: Optional[bool]) -> None:
        if self.number_of_subcomponents == 1:
            self._shared_yield = False
            self._has_fractions = False
            self._required_fraction_parameters = 0
        elif shared_yield is None:
            raise ValueError("Please specify whether yields are shared via the argument shared_yields "
                             "when the component consists of more than one template!")
        else:
            self._shared_yield = shared_yield
            if shared_yield:
                self.check_template_yield_indices()
                self._has_fractions = True
                self._required_fraction_parameters = self.number_of_subcomponents - 1
            else:
                self._has_fractions = False
                self._required_fraction_parameters = 0

    def initialize_parameters(
            self,
            fraction_parameters: Union[None, List[TemplateParameter], TemplateParameter]
    ) -> None:
        if not (isinstance(fraction_parameters, List) or fraction_parameters is None):
            raise ValueError("Expecting list of TemplateParameters or None for argument 'fraction_parameters'")
        if self._required_fraction_parameters > 0:
            if (fraction_parameters is None or (isinstance(fraction_parameters, List)
                                                and len(fraction_parameters) != self._required_fraction_parameters)):
                raise ValueError(f"The component {self.name}, consisting of {self.number_of_subcomponents} templates,"
                                 f"requires {self._required_fraction_parameters} ModelParameters of parameter_type "
                                 f"'fraction' to be set, but you provided "
                                 + "'None'!" if fraction_parameters is None else
                                 f"a list containing {len(fraction_parameters)} parameters!")
            assert isinstance(fraction_parameters, list)
            if not all(isinstance(fraction_parameter, TemplateParameter) for fraction_parameter in fraction_parameters):
                raise ValueError(f"Argument 'fraction_parameters' must be a list of TemplateParameters!\n"
                                 f"You provided a list containing the types "
                                 f"{[type(p) for p in fraction_parameters]}...")
            if not all(fraction_parameter.parameter_type == "fraction" for fraction_parameter in fraction_parameters):
                raise ValueError(f"Expected TemplateParameters of parameter_type 'fraction', "
                                 f"but fraction_parameters contains TemplateParameters of the parameter_types "
                                 f"{[p.parameter_type for p in fraction_parameters]}!")

            self._fraction_parameters = fraction_parameters
            for template, fraction_parameter in zip(self._templates[:-1], fraction_parameters):
                template.fraction_parameter = fraction_parameters
            # For the last template of the component the template.fraction_parameter remains unset.

        else:
            if not (fraction_parameters is None
                    or (isinstance(fraction_parameters, List) and len(fraction_parameters) == 0)):
                raise ValueError(f"The component does not require fraction TemplateParameters, but you provided some!")
            # For only one template or templates which do not share their yields,
            # the template.fraction_parameter remains unset.

    @property
    def name(self) -> str:
        return self._name

    @property
    def binning(self) -> Binning:
        return self._binning

    @property
    def params(self) -> ParameterHandler:
        return self._params

    @property
    def shared_yield(self) -> bool:
        return self._shared_yield

    @property
    def has_fractions(self) -> bool:
        return self._has_fractions

    @property
    def process_names(self) -> List[str]:
        return [t.process_name for t in self._templates]

    @property
    def component_serial_number(self) -> int:
        assert self._component_serial_number is not None
        return self._component_serial_number

    @component_serial_number.setter
    def component_serial_number(self, component_serial_number: int) -> None:
        if self._component_serial_number is not None:
            raise RuntimeError(f"Trying to reset component serial number "
                               f"from {self._component_serial_number} to {component_serial_number}!")
        self._component_serial_number = component_serial_number
        for template in self._templates:
            template.component_serial_number = component_serial_number

    @property
    def template_serial_numbers(self) -> Tuple[int, ...]:
        assert self._templates is not None
        return tuple(template.serial_number for template in self._templates)

    @property
    def component_index(self) -> int:
        assert self._component_index is not None
        return self._component_index

    @component_index.setter
    def component_index(self, index: int) -> None:
        self._parameter_setter_checker(parameter=self._component_index, parameter_name="component_index")
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._component_index = index
        for template in self._templates:
            template.component_index = index

    @property
    def channel_index(self) -> int:
        assert self._channel_index is not None
        return self._channel_index

    @channel_index.setter
    def channel_index(self, index: int) -> None:
        self._parameter_setter_checker(parameter=self._channel_index, parameter_name="channel_index")
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._channel_index = index
        for template in self._templates:
            template.channel_index = index

    @property
    def number_of_subcomponents(self) -> int:
        return len(self._templates)

    @property
    def required_fraction_parameters(self) -> int:
        return self._required_fraction_parameters

    @property
    def sub_templates(self) -> Tuple[Template]:
        return self._templates

    @property
    def data_column_names(self) -> Optional[List[str]]:
        if not all(t.data_column_names == self._templates[0].data_column_names if t.data_column_names is not None
                   else t.data_column_names is self._templates[0].data_column_names
                   for t in self._templates):
            raise RuntimeError(f"Inconsistency in data_column_names of component {self.name}:\n\t-"
                               + "\n\t-".join([f"{t.name}: {t.data_column_names}" for t in self._templates]))
        return self._templates[0].data_column_names

    def _parameter_setter_checker(self, parameter, parameter_name) -> None:
        if parameter is not None:
            name_info = "" if self.name is None else f" with name '{self.name}'"
            raise RuntimeError(f"Trying to reset {parameter_name} for component{name_info}.")

    def check_template_yield_indices(self):
        first_model_param = self._templates[0].yield_parameter.base_model_parameter
        if not all(first_model_param is t.yield_parameter.base_model_parameter for t in self._templates):
            raise RuntimeError(f"Trying to setup a component from multiple templates which should share their "
                               f"yield parameter for templates which do not have the same yield parameter!\n"
                               f"The used templates use the following yield model parameters:\n"
                               + "\n".join([f"{t.name}\n{t.yield_parameter.base_model_parameter.as_string()}"
                                            for t in self._templates])
                               )
