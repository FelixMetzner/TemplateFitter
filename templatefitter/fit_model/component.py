"""
The Component class is used to hold one or multiple templates which describe one component in a reconstruction channel.
If a component consists of multiple sub-components this class manages the respective fractions of the components.
Otherwise it acts just as a wrapper for the template class.
"""

import logging

from typing import Union, List

from templatefitter.fit_model.template import Template

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Component"]


class Component:
    def __init__(
            self,
            templates: Union[Template, List[Template]]
    ):
        if isinstance(templates, Template):
            self._templates = (templates,)
        elif isinstance(templates, list):
            if not all(isinstance(t, Template) for t in templates):
                raise ValueError("The parameter 'template' must be a Template or a List of Templates!\n"
                                 "You provided a list with the types:\n\t-"
                                 + "\n\t-".join([str(type(t)) for t in templates]))
            self._templates = tuple(t for t in templates)
        else:
            raise ValueError(f"The parameter 'template' must be a Template or a List of Templates!\n"
                             f"You provided an object of type {type(templates)}.")

    # TODO: needs method to get
    #           - templates
    #           - template fractions
    #           - indices
    #           - initial parameters
    #           - parameters ?

    # TODO: needs method to assign indices and parameters to the templates once the model is fixed.

    # TODO: Check that every template of a model uses the same ParameterHandler instance!
