from functools import partial
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from torch.amp import custom_fwd
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


class HWD_WaveletTransform(nn.Module):
    def __init__(self):
        super(HWD_WaveletTransform, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

    def forward(self, x):
        yL, yH = self.wt(x)
        yL_reshaped = yL.unsqueeze(2)
        x = torch.cat([yL_reshaped, yH[0]], dim=2)
        return x


class DFWConv_Haar(nn.Module):
    def __init__(self, in_channels, kernel_size=5, wt_type="haar"):
        super(DFWConv_Haar, self).__init__()
        self.in_channels = in_channels
        self.dilation = 1

        _, self.iwt_filter = create_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = HWD_WaveletTransform()
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        self.edge_extractor = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.low_freq_conv_branch = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding="same", stride=1, dilation=1,
            groups=in_channels, bias=False,
        )
        self.low_freq_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=0.1)
        self.high_freq_conv_branch = nn.Conv2d(
            in_channels * 3, in_channels * 3, kernel_size,
            padding="same", stride=1, dilation=1,
            groups=in_channels * 3, bias=False,
        )
        self.high_freq_scale = _ScaleModule([1, in_channels * 3, 1, 1], init_scale=0.1)
        self.DFWConv_fusion = AuraFuse()

    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, x):
        original_input = x
        curr_x_ll = x
        curr_shape = curr_x_ll.shape

        curr_x = self.wt_function(curr_x_ll)
        low_band = curr_x[:, :, 0:1, :, :].squeeze(2)
        high_bands = curr_x[:, :, 1:4, :, :]

        low_band = self.low_freq_conv_branch(low_band)
        low_band = self.low_freq_scale(low_band)
        low_band = low_band.unsqueeze(2)

        batch_size, channels, num_bands_high, height, width = high_bands.size()
        high_bands_flat = high_bands.reshape(batch_size, channels * num_bands_high, height, width)
        high_bands_flat = self.high_freq_conv_branch(high_bands_flat)
        high_bands_flat = self.high_freq_scale(high_bands_flat)
        high_bands = high_bands_flat.reshape(batch_size, channels, num_bands_high, height, width)

        curr_x_tag = torch.cat([low_band, high_bands], dim=2)

        x_ll = curr_x_tag[:, :, 0, :, :]
        x_h = curr_x_tag[:, :, 1:4, :, :]

        next_x_ll = 0
        curr_x_ll = x_ll
        curr_x_h = x_h
        curr_x_ll = curr_x_ll + next_x_ll
        curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
        next_x_ll = self.iwt_function(curr_x)
        next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        x_tag = next_x_ll
        edge_features = original_input - self.edge_extractor(original_input)
        edge_weights = self.weight_generator(edge_features)
        out = self.DFWConv_fusion(original_input, x_tag * edge_weights)
        return out