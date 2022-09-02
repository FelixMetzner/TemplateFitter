"""
This package provides
    - A FractionConversionManager Class to create and test the fraction conversion performed for the fit.
    - A generic FitObjectManager Class to manage Templates and Components.
"""
import logging

from templatefitter.fit_model.channel import ModelChannels, Channel
from typing import Optional, Union, List, NamedTuple, MutableMapping, TypeVar
from templatefitter.fit_model.template import Template
from templatefitter.fit_model.component import Component
from templatefitter.fit_model.parameter_handler import ParameterHandler, ModelParameter
from scipy.linalg import block_diag
import numpy as np

logging.getLogger(__name__).addHandler(logging.NullHandler())


class FractionConversionInfo(NamedTuple):
    needed: bool
    conversion_matrix: np.ndarray
    conversion_vector: np.ndarray


class FractionManager:
    def __init__(self, param_handler: ParameterHandler, channels: ModelChannels):

        self._param_handler = param_handler
        self._channels = channels
        self.fraction_conversion = None  # type: Optional[FractionConversionInfo]

    def convert_fractions(self):
        self._initialize_fraction_conversion()
        self._check_fraction_conversion()

    def check_matrix_shapes(self, yield_params: np.ndarray, fraction_params: np.ndarray) -> bool:
        assert self.fraction_conversion is not None
        assert len(fraction_params.shape) == 1, (len(fraction_params.shape), fraction_params.shape)
        assert (
            len(self.fraction_conversion.conversion_vector.shape) == 1
        ), self.fraction_conversion.conversion_vector.shape
        assert (
            len(self.fraction_conversion.conversion_matrix.shape) == 2
        ), self.fraction_conversion.conversion_matrix.shape
        assert yield_params.shape[0] == len(self.fraction_conversion.conversion_vector), (
            yield_params.shape[0],
            len(self.fraction_conversion.conversion_vector),
        )
        assert self.fraction_conversion.conversion_matrix.shape[0] == yield_params.shape[0], (
            self.fraction_conversion.conversion_matrix.shape[0],
            yield_params.shape[0],
        )
        assert self.fraction_conversion.conversion_matrix.shape[1] == fraction_params.shape[0], (
            self.fraction_conversion.conversion_matrix.shape[1],
            fraction_params.shape[0],
        )

        return True

    def _initialize_fraction_conversion(self) -> None:
        # Fraction conversion matrix and vector should be equal in all channels.
        # The matrices and vectors are generated for each channel, tested for equality and then stored once.
        assert self.fraction_conversion is None
        conversion_matrices = []  # type: List[np.ndarray]
        conversion_vectors = []  # type: List[np.ndarray]
        for channel in self._channels:
            matrices_for_channel = []  # type: List[np.ndarray]
            vectors_for_channel = []  # type: List[np.ndarray]
            for component in channel:
                n_sub = component.number_of_subcomponents
                if component.has_fractions:
                    matrix_part1 = np.diag(np.ones(n_sub - 1))
                    matrix_part2 = -1 * np.ones(n_sub - 1)
                    matrix = np.vstack([matrix_part1, matrix_part2])
                    matrices_for_channel.append(matrix)
                    vector = np.zeros((n_sub, 1))
                    vector[-1][0] = 1.0
                    vectors_for_channel.append(vector)
                else:
                    matrices_for_channel.append(np.zeros((n_sub, n_sub)))
                    vectors_for_channel.append(np.ones((n_sub, 1)))

            conversion_matrices.append(block_diag(*matrices_for_channel))
            conversion_vectors.append(np.vstack(vectors_for_channel))

        assert all(m.shape[0] == v.shape[0] for m, v in zip(conversion_matrices, conversion_vectors))
        assert all(
            m.shape[0] == n_f for m, n_f in zip(conversion_matrices, self._channels.number_of_independent_templates)
        )
        assert all(np.array_equal(m, conversion_matrices[0]) for m in conversion_matrices)
        assert all(np.array_equal(v, conversion_vectors[0]) for v in conversion_vectors)

        self.fraction_conversion = FractionConversionInfo(
            needed=(not all(conversion_vectors[0] == 1)),
            conversion_matrix=conversion_matrices[0],
            conversion_vector=conversion_vectors[0],
        )

    @property
    def _fractions_are_needed(self) -> bool:
        assert all(
            [
                n_it <= n_t
                for n_it, n_t in zip(self._channels.number_of_independent_templates, self._channels.number_of_templates)
            ]
        ), "\n".join(
            [
                f"{i}] <= {t}"
                for i, t in zip(self._channels.number_of_independent_templates, self._channels.number_of_templates)
            ]
        )
        return not (self._channels.number_of_independent_templates == self._channels.number_of_templates)

    def _check_fraction_conversion(self) -> None:

        assert self.fraction_conversion is not None
        assert self._fractions_are_needed == self.fraction_conversion.needed

        if self._fractions_are_needed:
            assert np.any(self.fraction_conversion.conversion_vector != 1), self.fraction_conversion.conversion_vector
            assert np.any(self.fraction_conversion.conversion_matrix != 0), self.fraction_conversion.conversion_matrix
            assert (
                len(self.fraction_conversion.conversion_vector.shape) == 1
            ), self.fraction_conversion.conversion_vector.shape
            assert self.fraction_conversion.conversion_vector.shape[0] == max(self._channels.number_of_templates), (
                self.fraction_conversion.conversion_vector.shape[0],
                max(self._channels.number_of_templates),
                self._channels.number_of_templates,
            )
            assert (
                len(self.fraction_conversion.conversion_matrix.shape) == 2
            ), self.fraction_conversion.conversion_matrix.shape
            assert self.fraction_conversion.conversion_matrix.shape[0] == max(self._channels.number_of_templates), (
                self.fraction_conversion.conversion_matrix.shape[0],
                max(self._channels.number_of_templates),
                self._channels.number_of_templates,
            )
            _first_channel = self._channels[0]
            assert isinstance(_first_channel, Channel), type(_first_channel).__name__
            assert self.fraction_conversion.conversion_matrix.shape[1] == len(_first_channel.fractions_mask), (
                self.fraction_conversion.conversion_matrix.shape[1],
                len(_first_channel.fractions_mask),
                _first_channel.fractions_mask,
            )
        else:
            logging.info(
                "Fraction parameters are not used, as no templates of the same channel share a common yield parameter."
            )
            assert np.all(self.fraction_conversion.conversion_vector == 1), self.fraction_conversion.conversion_vector
            assert np.all(self.fraction_conversion.conversion_matrix == 0), self.fraction_conversion.conversion_matrix

            assert all(sum(c.required_fraction_parameters) == 0 for c in self._channels), "\n".join(
                [f"{c.name}: {c.required_fraction_parameters}" for c in self._channels]
            )
            yields_i = self._param_handler.get_parameter_indices_for_type(
                parameter_type=ParameterHandler.yield_parameter_type
            )
            assert all(
                len(yields_i) >= c.total_number_of_templates for c in self._channels
            ), f"{len(yields_i)}\n" + "\n".join([f"{c.name}: {c.total_number_of_templates}" for c in self._channels])

    def check_fraction_parameters(self, model_parameters: List[ModelParameter]) -> bool:
        # Check number of fraction parameters
        assert self._channels.number_of_dependent_templates == self._channels.number_of_fraction_parameters, (
            self._channels.number_of_dependent_templates,
            self._channels.number_of_fraction_parameters,
        )
        frac_i = self._param_handler.get_parameter_indices_for_type(
            parameter_type=ParameterHandler.fraction_parameter_type
        )
        assert max(self._channels.number_of_dependent_templates) == len(frac_i), (
            f"Required fraction_parameters = "
            f"{max(self._channels.number_of_dependent_templates)}\n"
            f"Registered fraction model parameters = "
            f"{len(frac_i)}"
        )

        # Check that fraction parameters are the same for each channel
        assert all(
            nf == self._channels.number_of_fraction_parameters[0] for nf in self._channels.number_of_fraction_parameters
        ), self._channels.number_of_fraction_parameters

        fraction_parameter_infos = self._param_handler.get_parameter_infos_by_index(indices=frac_i)
        fraction_model_parameters = [model_parameters[fpi.model_index] for fpi in fraction_parameter_infos]

        # Check order and consistency of fraction parameters
        for channel in self._channels:
            par_i = 0
            comps_and_temps = [(c, t) for c in channel.components for t in c.sub_templates]

            assert len(comps_and_temps) == channel.total_number_of_templates, (
                len(comps_and_temps),
                channel.total_number_of_templates,
            )
            assert len(channel.fractions_mask) == channel.total_number_of_templates, (
                len(channel.fractions_mask),
                channel.total_number_of_templates,
            )

            last_mask_value = False
            for counter, ((component, template), mask_value) in enumerate(zip(comps_and_temps, channel.fractions_mask)):
                if mask_value:
                    assert component.has_fractions
                    assert fraction_parameter_infos[par_i].parameter_type == ParameterHandler.fraction_parameter_type, (
                        par_i,
                        fraction_parameter_infos[par_i].as_string(),
                    )
                    temp_serial_num = fraction_model_parameters[par_i].usage_serial_number_list[channel.channel_index]
                    assert temp_serial_num == template.serial_number, (temp_serial_num, template.serial_number)
                    temp_param = fraction_model_parameters[par_i].usage_template_parameter_list[channel.channel_index]
                    assert template.fraction_parameter == temp_param, (
                        template.fraction_parameter.as_string(),
                        temp_param.as_string(),
                    )

                elif (not mask_value) and last_mask_value:
                    assert component.has_fractions
                    assert component.template_serial_numbers[-1] == template.serial_number, (
                        component.template_serial_numbers[-1],
                        template.serial_number,
                    )
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

        return True


FitObject = TypeVar("FitObject", Template, Component)


class FitObjectManager(MutableMapping[Union[str, int], FitObject]):
    def __init__(self):
        super().__init__()

        self._fit_object_mapping = {}
        self._fit_objects = []

    def __getitem__(self, item: Union[str, int]) -> FitObject:
        if isinstance(item, int):
            return self._fit_objects[item]
        elif isinstance(item, str):
            return self._fit_objects[self._fit_object_mapping[item]]
        else:
            raise TypeError(f"Keys of FitObjectManager must be either int or str, not {type(item)}.")

    def __setitem__(self, index: Union[str, int], value: FitObject):
        raise Exception("FitObjectManager is append-only.")

    def __delitem__(self, item: Union[int, str]):
        raise Exception("FitObjectManager is append-only.")

    def __repr__(self) -> str:
        return repr(
            {
                (key, index): fit_object
                for fit_object, (key, index) in zip(self._fit_objects, self._fit_object_mapping.items())
            }
        )

    def __iter__(self):
        return iter(self._fit_objects)

    def get_index_of_fit_objects(self, fit_object: FitObject) -> int:
        if fit_object in self._fit_objects:
            return self._fit_object_mapping[fit_object.name]
        else:
            raise IndexError(
                f"The fit object with name {fit_object.name} has not"
                f" been registered in the FitObjectManager and no index can be returned."
            )

    def get_fit_objects_by_process_name(self, process_name: str) -> List[FitObject]:

        if hasattr(self._fit_objects[0], "process_name"):
            obj_list = [o for o in self._fit_objects if o.process_name == process_name]

            if not len(obj_list):
                available_processes = list(set([o.process_name for o in self._fit_objects]))
                raise RuntimeError(
                    f"No templates with process name '{process_name}' could be found.\n"
                    f"The following process names are registered:\n\t" + "\n\t- ".join(available_processes)
                )
        elif hasattr(self._fit_objects[0], "process_names"):
            obj_list = [o for o in self._fit_objects if process_name in o.process_names]
            if not len(obj_list):
                available_processes = list(set([pn for o in self._fit_objects for pn in o.process_names]))
                raise RuntimeError(
                    f"No components containing process name '{process_name}' could be found.\n"
                    f"The following process names are registered:\n\t" + "\n\t- ".join(available_processes)
                )

        return obj_list

    def append(self, fit_object: FitObject):

        if fit_object.name in self._fit_object_mapping:
            raise ValueError(
                f"FitObject with name {fit_object.name} already exists in FitObjectManager with the index "
                f"{self._fit_object_mapping[fit_object.name]}"
            )

        fit_object.serial_number = len(self._fit_objects)

        self._fit_object_mapping[fit_object.name] = len(self._fit_objects)
        self._fit_objects.append(fit_object)

    def __len__(self):
        return len(self._fit_objects)
