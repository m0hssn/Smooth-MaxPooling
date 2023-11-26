import torch
import torch.nn.functional as f


def smooth_MaxPool2d(input, kernel_size, stride, padding=0, ceil_mode=False, count_include_pad=True):
    return torch.log(f.avg_pool2d(input=torch.exp(input), kernel_size=kernel_size, stride=stride,
                                  padding=padding, ceil_mode=ceil_mode,
                                  count_include_pad=count_include_pad, divisor_override=1))


def smooth_MaxPool3d(input, kernel_size, stride, padding=0, ceil_mode=False, count_include_pad=True):
    return torch.log(f.avg_pool3d(input=torch.exp(input), kernel_size=kernel_size, stride=stride,
                                  padding=padding, ceil_mode=ceil_mode,
                                  count_include_pad=count_include_pad, divisor_override=1))


def smooth_MaxPool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad):
    return torch.log(kernel_size * f.avg_pool1d(input=torch.exp(input), kernel_size=kernel_size, stride=stride,
                                                padding=padding, ceil_mode=ceil_mode,
                                                count_include_pad=count_include_pad))
