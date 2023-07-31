import functools
import warnings
from typing import Union, Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.utils.data
import torchvision.transforms as vision_transforms
import torchvision.transforms.functional as vision_transform_fnc

IntOrIntTuple = Union[int, Tuple[int, ...]]
NDArray = np.ndarray


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    >>> @deprecated
    >>> def my_func(*args, **kwargs):
    >>>     return args, kwargs
    >>> my_func(1, 2, a=3, b=4)
    warning: Call to deprecated function my_func.
    (1, 2), {'a': 3, 'b': 4}
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def shape_after_conv(ndim_shape: NDArray,
                     kernel_size: NDArray,
                     padding: NDArray = np.array([0]),
                     stride: NDArray = np.array([1]),
                     dilation: NDArray = np.array([1])) -> NDArray:
    """
    Example:
    For a date in 2 dimension:
        ndim_shape:[28, 28],
    kernel size shall be an array, even the kernel is a Square, shape=(3, 3), as well as padding, string, dilation
    :param ndim_shape:
    :param kernel_size:
    :param padding:
    :param stride:
    :param dilation:
    :return:
    """
    if not np.all(isinstance(i, np.ndarray) for i in [ndim_shape, kernel_size, padding, stride, dilation]):
        raise TypeError(f"Function args shall be a instance of np.ndarray, but:\n"
                        f"ndim_shape is {type(ndim_shape)}, "
                        f"kernel_size is {type(kernel_size)}\n"
                        f"padding is {type(padding)}, "
                        f"stride is {type(stride)}\n"
                        f"dilation is {type(dilation)}\n")
    return ((ndim_shape + (2 * padding) - (dilation * (kernel_size - 1)) - 1) // stride) + 1


def shape_after_conv_transpose(
        ndim_shape: NDArray,
        kernel_size: NDArray,
        padding: NDArray = np.array([0]),
        stride: NDArray = np.array([0]),
        dilation: NDArray = np.array([1]), ) -> NDArray:
    if not np.all(isinstance(i, np.ndarray) for i in [ndim_shape, kernel_size, padding, stride, dilation]):
        raise TypeError(f"Function args shall be a instance of np.ndarray, but:\n"
                        f"ndim_shape is {type(ndim_shape)}, "
                        f"kernel_size is {type(kernel_size)}\n"
                        f"padding is {type(padding)}, "
                        f"stride is {type(stride)}\n"
                        f"dilation is {type(dilation)}\n")
    return (ndim_shape - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + padding + 1


def shape_after_n_time_convolution(
        ndim_shape: NDArray,
        kernel_size: NDArray,
        n_time: int = 1,
        padding: NDArray = np.array([0]),
        stride: NDArray = np.array([1]),
        dilation: NDArray = np.array([1]),

) -> NDArray:
    ret = ndim_shape
    for i in range(n_time):
        ret = shape_after_conv(
            ret,
            kernel_size=kernel_size[i],
            padding=padding[i],
            stride=stride[i],
            dilation=dilation[i]
        )
    return ret


def shape_after_n_time_convolution_transpose(
        ndim_shape: NDArray,
        kernel_size: NDArray,
        n_time: int = 1,
        padding: NDArray = np.array([0]),
        stride: NDArray = np.array([1]),
        dilation: NDArray = np.array([1]),

) -> NDArray:
    ret = ndim_shape
    for i in range(n_time):
        ret = shape_after_conv_transpose(
            ret,
            kernel_size=kernel_size[i],
            padding=padding[i],
            stride=stride[i],
            dilation=dilation[i]
        )
    return ret


def plot_waveform(waveform: Union[torch.Tensor, NDArray],
                  sample_rate: int,
                  title: str = "Waveform") -> None:
    """
    Plot waveform, where waveform is a tensor, where shape[0] is the number of channels
    and shape[1] is the number of frames.
    >>> import torchaudio
    >>> wave_form, sample_rate = torchaudio.load("test.wav")
    >>> plot_spectrogram(wave_form, sample_rate)
    :param title:
    :param waveform:
    :param sample_rate:
    :return:
    """
    waveform_np = waveform.numpy()
    num_channels, n_frame = waveform_np.shape
    time_axis = torch.arange(0, n_frame) / sample_rate  # time axis in seconds

    fig, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform_np[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_title(f"Channel {c + 1}")
        fig.suptitle(title)
        plt.show()


def plot_spectrogram(waveform: Union[torch.Tensor, NDArray],
                     sample_rate: int,
                     title: str = "Spectrogram") -> None:
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)

    if num_channels == 1:
        axes = [axes]

    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")

    figure.suptitle(title)
    plt.show(block=False)


def create_mask(size: Union[torch.Tensor, torch.Size, List[int]],
                mask_rate: float = 0.5) -> torch.Tensor:
    mask = torch.randn(*size) > mask_rate
    return mask


def create_mask_chunk_2d(size: torch.Size,
                         mask_rate: float = 0.5, b: int = 8) -> torch.Tensor:
    """
    1. Create a mini mask with the given (size // b)
        when b = 2, size=(4*4), mini_mask be-like:
        [[1 0]
         [0 1]]
    2. resize mini mask, up sample it to the size, with mode nearest
        [[1 0]       [[1 1 0 0]
         [0 1]]  -->  [1 1 0 0]
                      [0 0 1 1]
                      [0 0 1 1]]
    3. return the mask
    :param size:
    :param mask_rate:
    :param b:
    :return:
    """
    mini_mask_size: torch.Tensor = torch.div(torch.tensor(size, dtype=torch.int), b, rounding_mode="floor")
    mini_mask: torch.Tensor = create_mask(mini_mask_size, mask_rate)
    size: list = [*size]
    mask: torch.Tensor = vision_transform_fnc.resize(img=mini_mask.unsqueeze(0), size=size,
                                                     interpolation=vision_transforms.InterpolationMode.NEAREST)
    return mask.squeeze(0)


def tensor_masking(tensor_to_mask: torch.Tensor,
                   mask_rate: float = .5, mask: Optional = None) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a tensor of audio data, return a tensor of masked audio data, a tensor of unmasked audio, and the mask:

    Mask: [True, False, True, False, False, True, False, True, False, True]
    Mask to bit:    [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    Original audio: [1, 1, 4, 5, 1, 4, 1, 9, 1, 9]
    In this case, the masked audio will be:
                    [1, 0, 4, 0, 0, 4, 0, 9, 0, 9]
    And the Unmasked audio will be:
                    [0, 1, 0, 5, 1, 0, 1, 0, 1, 0]
    Masked audio + Unmasked audio = Original audio
    :param tensor_to_mask:
    :param mask_rate: Masking rate
    :param mask: Mask to use. If None, a random mask will be created.
    :return:
    """
    if mask is None:
        mask = create_mask(tensor_to_mask.shape, mask_rate)
    return torch.tensor(np.where(mask, tensor_to_mask.numpy(), 0)), \
        torch.tensor(np.where(~mask, tensor_to_mask.numpy(), 0)), mask
