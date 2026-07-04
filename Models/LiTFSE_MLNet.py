import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqSE2d(nn.Module):
    """
    Frequency-axis SE attention module
    """

    def __init__(self, freq_bins, reduction=8):
        super().__init__()
        mid_channels = max(freq_bins // reduction, 4)
        self.fc1 = nn.Linear(freq_bins, mid_channels)
        self.fc2 = nn.Linear(mid_channels, freq_bins)

    def forward(self, x):
        squeeze = x.mean(dim=(1, 3))
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        self.last_weights = excitation.detach()
        scale = excitation.unsqueeze(1).unsqueeze(3)
        return x * scale


class Backbone2DCNN(nn.Module):
    """
    2D CNN backbone for STFT spectrogram input (v3).
    Input: (B, C, freq_bins, time_frames) STFT spectrogram
    Output: (B, out_channels) classification logits
    """

    def __init__(self, in_channels=3, out_channels=9, freq_bins=129, use_freqse=None, **kwargs):
        super().__init__()

        if use_freqse is None:
            use_freqse = [1, 1, 1, 1]
        use_freqse = [int(x) for x in use_freqse[:4]]
        self.use_freqse = use_freqse

        conv1_layers = [
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        if use_freqse[0] == 1:
            conv1_layers.append(FreqSE2d(freq_bins, reduction=8))
        self.conv1 = nn.Sequential(*conv1_layers)

        conv2_layers = [
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        ]
        if use_freqse[1] == 1:
            conv2_layers.append(FreqSE2d(freq_bins // 2, reduction=8))
        self.conv2 = nn.Sequential(*conv2_layers)

        conv3_layers = [
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        ]
        if use_freqse[2] == 1:
            conv3_layers.append(FreqSE2d(freq_bins // 4, reduction=8))
        self.conv3 = nn.Sequential(*conv3_layers)

        conv4_layers = [
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        ]
        if use_freqse[3] == 1:
            conv4_layers.append(FreqSE2d(freq_bins // 8, reduction=8))
        self.conv4 = nn.Sequential(*conv4_layers)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.out(x)
        return x


class LiTFSE_MLNet(nn.Module):
    """
    LiTFSE-Net: Lightweight Frequency-Domain Multi-Label Attention Network
    """

    def __init__(self, in_channels=3, out_channels=9, kernel_size=11, **kwargs):
        super().__init__()
        self.n_fft = 256  # Frequency resolution = 12800/256 = 50Hz
        self.hop_length = 32
        self.freq_bins = self.n_fft // 2 + 1
        self.freq_shift_range = 10
        self.register_buffer("window", torch.hann_window(self.n_fft))

        self.backbone = Backbone2DCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            freq_bins=self.freq_bins,
        )

    def forward(self, x):
        B, C, T = x.shape

        stft = torch.stft(
            x.reshape(B * C, T), n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True
        )
        mag = torch.abs(stft)
        mag = torch.log1p(mag)
        time_frames = mag.shape[2]

        mag = mag.view(B, C, self.freq_bins, time_frames)

        out = self.backbone(mag)
        return out


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LiTFSE_MLNet(in_channels=3, out_channels=9).to(device)
    for T in [2048, 4096, 8192]:
        x = torch.randn(4, 3, T).to(device)
        y = model(x)
        print(f"Input: {x.shape} -> Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print("\nFreqSE2d visualization: see comments in code for usage.")
