import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        main_path = F.relu(self.bn1(self.conv1(x)))
        main_path = self.bn2(self.conv2(main_path))
        x = main_path + self.shortcut(x)
        return F.relu(x)
    


class DrumCRNN(nn.Module):
    def __init__(self, num_classes=8, rnn_hidden_size=128, freq_bins=256):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res_block1 = ResidualBlock(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2)) # freq/2, time/2

        self.res_block2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2)) # freq/4, time/4
        
        self.dropout = nn.Dropout(0.3)
        
        self.channel_reducer = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1)
        self.reducer_bn = nn.BatchNorm2d(16)

        # RNN 입력 크기는 풀링 2번에 맞게 그대로 유지
        num_channel = 16
        rnn_input_size = num_channel * (freq_bins // 4)
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.3
        )

        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        # 입력 shape: (batch, 1, freq, time)

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.res_block2(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = F.relu(self.reducer_bn(self.channel_reducer(x))) 

        x = x.permute(0, 3, 1, 2) # (batch, time, channels, freq)
        x = x.flatten(start_dim=2) # (batch, time, channels * freq)

        x, _ = self.rnn(x)

        x = self.fc(x)

        return x