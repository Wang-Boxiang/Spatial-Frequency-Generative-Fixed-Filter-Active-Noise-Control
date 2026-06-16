import torch
import torch.nn as nn
from typing import List, Optional


class CRNN_Multi(nn.Module):
    """Multi-task CRNN for source-grid classification and filter-weight regression."""

    def __init__(
        self,
        in_channels: int = 8,
        num_distance: int = 4,
        num_elevation: int = 3,
        num_azimuth: int = 6,
        reg_dim: int = 8,
        cnn_channels: Optional[List[int]] = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        bidirectional: bool = False,
        fc_dim: int = 96,
        dropout: float = 0.2,
        freq_bins: int = 513,
    ):
        super().__init__()
        del dropout

        if cnn_channels is None:
            cnn_channels = [16, 32, 64]

        self.freq_after = freq_bins // 8

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.pool_freq = nn.AdaptiveAvgPool2d((1, None))
        self.pool_time_reg = nn.AdaptiveAvgPool2d((None, 1))

        self.gru = nn.GRU(
            input_size=cnn_channels[2],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * factor, fc_dim),
            nn.ReLU(inplace=True),
        )

        self.head_d = nn.Linear(fc_dim, num_distance)
        self.head_el = nn.Linear(fc_dim, num_elevation)
        self.head_az = nn.Linear(fc_dim, num_azimuth)

        flat_dim = cnn_channels[2] * self.freq_after
        self.head_reg = nn.Sequential(
            nn.Linear(flat_dim, reg_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        conv_feat = self.conv3(self.conv2(self.conv1(x)))

        cls_feat = self.pool_freq(conv_feat).squeeze(2).permute(0, 2, 1)
        _, h_n = self.gru(cls_feat)
        if self.gru.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]

        feat = self.fc(h)
        out_distance = self.head_d(feat)
        out_elevation = self.head_el(feat)
        out_azimuth = self.head_az(feat)

        reg_feat = self.pool_time_reg(conv_feat).squeeze(3).flatten(1)
        out_regression = self.head_reg(reg_feat)

        return out_distance, out_elevation, out_azimuth, out_regression
