"""
The Component class is used to hold one or multiple templates which describe one component in a reconstruction channel.
If a component consists of multiple sub-components this class manages the respective fractions of the components.
Otherwise it acts just as a wrapper for the template class.
"""

import logging

from typing import Union, Optional, List, Tuple

from templatefitter.fit_model.template import Template
from templatefitter.binned_distributions.binning import Binning
from templatefitter.fit_model.parameter_handler import ParameterHandler

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Component"]


class Component:
    def __init__(
            self,
            templates: Union[Template, List[Template]],
            params: ParameterHandler,
            name: Optional[str] = None,
            shared_yield: bool = True,
            initial_fractions: Optional[Tuple[float]] = None
    ):
        self._binning = None
        self._templates = None
        self._initial_fractions = None
        self._fractions = None

        self._name = name
        self._params = params
        self._shared_yield = shared_yield
        self._has_fractions = False

        self._template_indices = None
        self._component_index = None
        self._channel_index = None

        self._initialize_templates(templates=templates)
        self._initialize_fractions(initial_fractions=initial_fractions)

    def _initialize_templates(self, templates: Union[Template, List[Template]]) -> None:
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
            self._templates = tuple(t for t in templates)
            for template_index, template in enumerate(templates):
                template.template_index = template_index
            self._binning = templates[0].binning
        else:
            raise ValueError(f"The parameter 'template' must be a Template or a List of Templates!\n"
                             f"You provided an object of type {type(templates)}.")

        if not all(t.params is self._params for t in self._templates):
            raise RuntimeError("The used ParameterHandler instances are not the same!")

    def _initialize_fractions(self, initial_fractions: Optional[Tuple[float]]) -> None:
        if initial_fractions is not None:
            if not len(initial_fractions) == len(self._templates):
                raise ValueError(f"Number of templates and number of initial fraction values must be equal.\n"
                                 f"You provided {len(self._templates)} templates and "
                                 f"{len(initial_fractions)} initial fraction values.")
            self._initial_fractions = initial_fractions
        elif len(self._templates) == 1:
            self._initial_fractions = (1.,)
        else:
            raise RuntimeError("The component consists of more than one template,"
                               "but no initial fractions have been provided for the component!"
                               )
        if self.shared_yield and len(self._templates) > 1:
            self._has_fractions = True

        self._fractions = self._initial_fractions

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
    def template_indices(self) -> List[int]:
        assert self._template_indices is not None
        return self._template_indices

    @template_indices.setter
    def template_indices(self, indices: Union[int, List[int]]) -> None:
        self._parameter_setter_checker(parameter=self._template_indices, parameter_name="template_indices")
        if isinstance(indices, int):
            self._template_indices = (indices,)
        elif isinstance(indices, list):
            if not all(isinstance(i, int) for i in indices):
                raise ValueError("Expected integer or list of integers...")
            self._template_indices = indices
        else:
            raise ValueError("Expected integer or list of integers...")

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
            template._component_index = index

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

    def _parameter_setter_checker(self, parameter, parameter_name) -> None:
        if parameter is not None:
            name_info = "" if self.name is None else f" with name '{self.name}'"
            raise RuntimeError(f"Trying to reset {parameter_name} for component{name_info}.")
