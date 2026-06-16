import torch
import torch.nn as nn

# 现在是Dropout after the head (on logits)，应该Dropout before final head

class CRNN_Multi(nn.Module):
    """
    CRNN model with 3 convolutional layers, 2-layer GRU, and multi-head outputs.
    Input shape: (batch, channel, frequency, time)
    Outputs: distance, elevation, azimuth classification logits and regression output.
    """
    def __init__(
        self,
        in_channels: int = 8,
        num_distance: int = 4,
        num_elevation: int = 3,
        num_azimuth: int = 6,
        reg_dim: int = 8,
        cnn_channels: list[int] = [16, 32, 64],
        gru_hidden: int = 128,
        gru_layers: int = 2,
        bidirectional: bool = False,
        fc_dim: int = 96,
        dropout: float = 0.2,
        freq_bins: int = 513,  # 原始频率维度
    ):
        super().__init__()
        # 三次池化后频率维度
        self.freq_after = freq_bins // (2 * 2 * 2)

        # --- 3-layer CNN ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[0]),
            # nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # reduce freq & time by 2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[1]),
            # nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[2]),
            # nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        # 用于分类的频率池化
        self.pool_freq = nn.AdaptiveAvgPool2d((1, None))
        self.pool_time_reg = nn.AdaptiveAvgPool2d((None, 1))  # keep freq, pool time -> 1


        # --- GRU for classification ---
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
        # self.dropout = nn.Dropout(dropout)
        self.head_d  = nn.Linear(fc_dim, num_distance)
        self.head_el = nn.Linear(fc_dim, num_elevation)
        self.head_az = nn.Linear(fc_dim, num_azimuth)

        # --- Regression head: explicit Linear ---
        flat_dim = cnn_channels[2] * self.freq_after
        self.head_reg = nn.Sequential(
            nn.Linear(flat_dim, reg_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, channel, freq, time)
        # --- shared conv backbone ---
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        conv_feat = x  # (batch, C, F', T')

        # --- classification branch ---
        cls = self.pool_freq(conv_feat).squeeze(2).permute(0, 2, 1)
        out_gru, h_n = self.gru(cls)
        if self.gru.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        feat = self.fc(h)
        # out_distance  = self.dropout(self.head_d(feat))
        # out_elevation = self.dropout(self.head_el(feat))
        # out_azimuth   = self.dropout(self.head_az(feat))
        out_distance  = self.head_d(feat)
        out_elevation = self.head_el(feat)
        out_azimuth   = self.head_az(feat)

        # --- regression branch: average over time, keep freq ---

        reg_feat = self.pool_time_reg(conv_feat).squeeze(3)  # (batch, C, F')
        reg_feat = reg_feat.flatten(1)       # (batch, C*F')
        out_regression = self.head_reg(reg_feat)

        return out_distance, out_elevation, out_azimuth, out_regression
