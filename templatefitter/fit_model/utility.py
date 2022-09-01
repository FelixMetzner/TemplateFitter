"""
Utility functions
"""

import numpy as np
from functools import cached_property

__all__ = ["pad_sequences", "check_bin_count_shape"]


def pad_sequences(
    sequences,
    maxlen=None,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0.0,
):
    """
    Taken as is from from keras.preprocessing.sequence, because we do not need keras for anything else:

    Pads each sequence to the same length (length of the longest sequence).

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)

    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    lengths = []
    for x in sequences:
        if not hasattr(x, "__len__"):
            raise ValueError("`sequences` must be a list of iterables. " "Found non-iterable: " + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                "Shape of sample %s of sequence at position %s is different from expected shape %s"
                % (trunc.shape[1:], idx, sample_shape)
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def check_bin_count_shape(bin_count: np.ndarray, number_of_channels: int, max_number_of_bins: int, where: str) -> None:
    assert bin_count is not None, where
    assert len(bin_count.shape) == 2, (where, bin_count.shape, len(bin_count.shape))
    assert bin_count.shape[0] == number_of_channels, (
        where,
        bin_count.shape,
        bin_count.shape[0],
        number_of_channels,
    )
    assert bin_count.shape[1] == max_number_of_bins, (
        where,
        bin_count.shape,
        bin_count.shape[1],
        max_number_of_bins,
    )


class immutable_cached_property(cached_property):
    def __set__(self, instance, value):
        raise AttributeError(f"{self.func.__name__} is a read-only attribute")

    def __delete__(self, instance):
        raise AttributeError(f"{self.func.__name__} is a read-only attribute")
