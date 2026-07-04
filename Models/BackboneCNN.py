from torch import nn
import torch
from utils.summary import summary


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, kernel_size=15, **kwargs):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=kernel_size, bias=True), nn.BatchNorm1d(16), nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, bias=True), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 64), nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        if len(x.shape) == 4:
            x = torch.squeeze(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = CNN()
    info = summary(model, (1, 1024), batch_size=-1, device="cpu")
    print(info)
