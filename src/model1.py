import torch
import torch.nn as nn
import torch.nn.functional as F

class DrumCRNN(nn.Module):
    
    def __init__(self, num_classes=8, rnn_hidden_size=32, freq_bins=256):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2)) # freq/2, time/2

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2)) # freq/4, time/4
        
        # ★★★ 드롭아웃 레이어 정의 ★★★
        self.dropout = nn.Dropout(0.3) # 비율은 0.3 ~ 0.5 사이에서 조절 가능
        
        self.depthwise_freq_conv = nn.Conv2d(64, 64, kernel_size=(freq_bins // 4, 1), padding=0, groups=64)
        self.depthwise_bn = nn.BatchNorm2d(64) # BatchNorm 채널 수도 64

        rnn_input_size = 64
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )

        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    
    def forward(self, x):
        # 입력 shape: (batch, 1, freq, time)

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        
        x = self.dropout(x) 

        x = F.relu(self.depthwise_bn(self.depthwise_freq_conv(x))) # (batch, 64, 1, time)

        x = x.squeeze(2) # (batch, 64, time)
        x = x.permute(0, 2, 1) # (batch, time, 64)

        x, _ = self.rnn(x)

        x = self.fc(x)

        return x