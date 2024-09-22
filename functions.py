import torch
import torch.nn as nn

class SmoothMaxPool1D(nn.Module):
    def __init__(self, kernel_size, stride, temperature=0.01):
        super(SmoothMaxPool1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.temperature = temperature

    def forward(self, x):
        batch_size, channels, length = x.size()
        out_length = (length - self.kernel_size) // self.stride + 1
        
        unfolded = x.unfold(2, self.kernel_size, self.stride)
        output = torch.logsumexp(unfolded / self.temperature, dim=-1) * self.temperature
        
        return output

class SmoothMaxPool2D(nn.Module):
    def __init__(self, kernel_size, stride, temperature=0.01):
        super(SmoothMaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.temperature = temperature

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        unfolded = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        unfolded = unfolded.contiguous().view(batch_size, channels, out_height, out_width, self.kernel_size, self.kernel_size)
        
        output = torch.logsumexp(unfolded / self.temperature, dim=(-2, -1)) * self.temperature
        
        return output

class SmoothMaxPool3D(nn.Module):
    def __init__(self, kernel_size, stride, temperature=0.01):
        super(SmoothMaxPool3D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.temperature = temperature

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        out_depth = (depth - self.kernel_size) // self.stride + 1
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        unfolded = x.unfold(2, self.kernel_size, self.stride) \
                   .unfold(3, self.kernel_size, self.stride) \
                   .unfold(4, self.kernel_size, self.stride)
        unfolded = unfolded.contiguous().view(batch_size, channels, out_depth, out_height, out_width,
                                               self.kernel_size, self.kernel_size, self.kernel_size)
        
        output = torch.logsumexp(unfolded / self.temperature, dim=(-2, -1, -3)) * self.temperature
        
        return output
