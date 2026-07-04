from torch import nn
import torch
import numpy as np
import random
import os
import torch.nn.init as init
import math
import torch.nn.functional as F

# random seed
seed = 88
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# set the min and max frequency of the kernel function
fmin = 0.03
fmax = 0.45
random_scale = 2e-3


# traditional convolution layer with Real-Imaginary mechanism
class BaseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.stride = stride
        self.padding = padding

        self.phases = ["real", "imag"]
        self.weight = torch.Tensor(len(self.phases), out_channels, in_channels, self.kernel_size)
        if bias:
            self.bias = torch.Tensor(len(self.phases), out_channels)
        else:
            self.bias = None

        for phase in self.phases:
            self.weight[self.phases.index(phase)] = torch.Tensor(out_channels, in_channels, self.kernel_size)
            init.kaiming_uniform_(self.weight[self.phases.index(phase)], a=math.sqrt(5))
            if bias:
                self.bias[self.phases.index(phase)] = torch.Tensor(out_channels)
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[self.phases.index(phase)])
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias[self.phases.index(phase)], -bound, bound)

        if self.__class__.__name__ == "BaseConv1d":
            self.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias)

    def forward(self, input):
        result = {}
        for phase in self.phases:
            if self.bias is None:
                result[phase] = F.conv1d(
                    input, self.weight[self.phases.index(phase)], stride=self.stride, padding=self.padding
                )
            else:
                result[phase] = F.conv1d(
                    input,
                    self.weight[self.phases.index(phase)],
                    bias=self.bias[self.phases.index(phase)],
                    stride=self.stride,
                    padding=self.padding,
                )
        output = torch.sqrt(result[self.phases[0]].pow(2) + result[self.phases[1]].pow(2) + 1e-8)
        output = torch.clamp(output, max=1e4)
        return output


# prepare for kernel function by adding limits and rewrite forward function
class BaseFuncConv1d(BaseConv1d):
    def __init__(self, *pargs, **kwargs):
        kwargs_new = {}
        for k in kwargs.keys():
            if k in ["in_channels", "out_channels", "kernel_size", "stride", "padding", "bias"]:
                kwargs_new[k] = kwargs[k]
        super().__init__(*pargs, **kwargs_new)
        if self.__class__.__name__ == "BaseFuncConv1d":
            self.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias)
            self.superparams = self.weight

    def _clamp_parameters(self):
        with torch.no_grad():
            for i in range(len(self.params_bound)):
                self.superparams.data[:, :, i].clamp_(self.params_bound[i][0], self.params_bound[i][1])

    def WeightForward(self):
        if self.clamp_flag:
            self._clamp_parameters()
        l00 = []
        for phase in self.phases:
            l0 = []
            for i in range(self.superparams.shape[0]):
                l1 = []
                for j in range(self.superparams.shape[1]):
                    l1.append(self.weightforward(self.kernel_size, self.superparams[i, j], phase).unsqueeze(0))
                l0.append(torch.vstack(l1).unsqueeze(0))
            l00.append(torch.vstack(l0).unsqueeze(0))
            self.weight = torch.vstack(l00)

    def forward(self, input):
        if self.__class__.__name__ != "BaseFuncConv1d":
            self.WeightForward()
        return super().forward(input)


# kernel functions have differences in the weightforward function and the _reset_parameters function
class TFconv_STTF(BaseFuncConv1d):  # kernel_size = out_channels * 2 - 1
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        clamp_flag=True,
        params_bound=((0, 0.5),),
        init_freqs=None,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound
        self.init_freqs = init_freqs
        self.superparams = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, len(params_bound)))
        self._reset_parameters()
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)

    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape  # (out_channels, in_channels)
            if self.init_freqs is not None:
                init_freqs_tensor = torch.tensor(self.init_freqs, dtype=torch.float32)
                if init_freqs_tensor.numel() == shape[0]:
                    temp0 = init_freqs_tensor.unsqueeze(1).expand(shape[0], shape[1])
                else:
                    temp0 = init_freqs_tensor.reshape(shape)
            else:
                fmin, fmax = 0.03, 0.45
                temp0 = torch.linspace(fmin, fmax, shape.numel()).reshape(shape)
            self.superparams.data[:, :, 0] = temp0
            self.WeightForward()

    def weightforward(self, lens, params, phase):
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())
        T = torch.arange(-(lens // 2), lens - (lens // 2)).to(params.device)
        sigma = torch.tensor(0.52).to(params.device)
        if self.phases.index(phase) == 0:
            result = torch.exp(-(T / (lens // 2)).pow(2) / sigma.pow(2) / 2) * torch.cos(2 * math.pi * params[0] * T)
        else:
            result = torch.exp(-(T / (lens // 2)).pow(2) / sigma.pow(2) / 2) * torch.sin(2 * math.pi * params[0] * T)
        return result


class TFconv_Chirplet(BaseFuncConv1d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, clamp_flag=True, init_freqs=None
    ):
        max_t = kernel_size // 2
        params_bound = ((0, 0.5), (-1 / max_t**2, 1 / max_t**2))
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound
        self.init_freqs = init_freqs
        self.superparams = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, len(params_bound)))
        self._reset_parameters()
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)

    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape

            if self.init_freqs is not None:
                init_freqs_tensor = torch.tensor(self.init_freqs, dtype=torch.float32)
                if init_freqs_tensor.numel() == shape[0]:
                    temp0 = init_freqs_tensor.unsqueeze(1).expand(shape[0], shape[1])
                else:
                    temp0 = init_freqs_tensor.reshape(shape)
            else:
                fmin, fmax = 0.03, 0.45
                temp0 = torch.linspace(fmin, fmax, shape.numel()).reshape(shape)

            self.superparams.data[:, :, 0] = temp0
            self.superparams.data[:, :, 1].normal_(0, 1e-4)
            self.WeightForward()

    def weightforward(self, lens, params, phase):
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())
        T = torch.arange(-(lens // 2), lens - (lens // 2)).to(params.device)
        sigma = torch.tensor(0.52).to(params.device)
        if self.phases.index(phase) == 0:
            result = torch.exp(-(T / (lens // 2)).pow(2) / sigma.pow(2) / 2) * torch.cos(
                2 * math.pi * (params[1] / 2 * T.pow(2) + params[0] * T)
            )
        else:
            result = torch.exp(-(T / (lens // 2)).pow(2) / sigma.pow(2) / 2) * torch.sin(
                2 * math.pi * (params[1] / 2 * T.pow(2) + params[0] * T)
            )
        return result


class TFconv_Morlet(BaseFuncConv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        clamp_flag=True,
        params_bound=((0.4, 10),),
        init_freqs=None,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound
        self.init_freqs = init_freqs
        self.superparams = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, len(params_bound)))
        self._reset_parameters()
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)

    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape  # (out_channels, in_channels)
            if self.init_freqs is not None:
                init_scale = torch.tensor(self.init_freqs, dtype=torch.float32)
                if init_scale.numel() == shape[0]:
                    temp_s = init_scale.unsqueeze(1).expand(shape[0], shape[1])
                else:
                    temp_s = init_scale.reshape(shape)
            else:
                scale_min, scale_max = self.params_bound[0]  # e.g., (0.4, 10)
                log_min = np.log2(scale_min)
                log_max = np.log2(scale_max)
                log_vals = torch.linspace(log_min, log_max, shape.numel())
                temp_s = torch.pow(2.0, log_vals).reshape(shape)
            self.superparams.data[:, :, 0] = temp_s
            self.WeightForward()

    def weightforward(self, lens, params, phase):
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())
        T = torch.arange(-(lens // 2), lens - (lens // 2)).to(params.device)
        T = T / params[0]
        sigma = torch.tensor(0.6).to(params.device)
        fc_len = int(self.out_channels)
        if self.phases.index(phase) == 0:
            result = (
                params[0].pow(-1)
                * torch.exp(-(T / fc_len).pow(2) / sigma.pow(2) / 2)
                * torch.cos(2 * math.pi * 0.2 * T)
            )
        else:
            result = (
                params[0].pow(-1)
                * torch.exp(-(T / fc_len).pow(2) / sigma.pow(2) / 2)
                * torch.sin(2 * math.pi * 0.2 * T)
            )
        return result


if __name__ == "__main__":
    for item in [TFconv_STTF, TFconv_Chirplet, TFconv_Morlet]:
        print(item.__name__)
        model = TFconv_STTF(1, 8, 15, padding=7, stride=1, bias=False, clamp_flag=True)
        input = torch.randn([1, 1, 1024])
        out = model(input)
        out.mean().backward()
        print(out.shape)
        print(f"{item.__name__:s} test pass")
