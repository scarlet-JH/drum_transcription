import torch
import torch.nn as nn
import torch.nn.functional as F

class DrumCRNN(nn.Module):
    def __init__(self, num_classes=8, rnn_hidden_size=128, freq_bins=128):
        super(DrumCRNN, self).__init__()

        # ★★★ 수정: conv3, bn3 레이어 다시 추가 ★★★
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(0.25)
        
        # ★★★ 수정: 마지막 Conv 레이어의 채널(64)을 입력받도록 변경 ★★★
        self.channel_reducer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        # RNN 입력 크기는 풀링 2번에 맞게 그대로 유지
        rnn_input_size = freq_bins // 4
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 1, freq_bins, time_steps)
        
        # --- 풀링 2번 ---
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # ★★★ 수정: conv3는 풀링 없이 적용 ★★★
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.dropout(x)
        # 현재 x shape: (batch, 64, freq_bins/4, time_steps/4)
        
        # 1x1 Conv로 채널 압축
        x = self.channel_reducer(x)

        # 1번째 차원의 크기가 1이면, 해당차원 제거
        x = x.squeeze(1)

        # (Batch, Freq, Time) -> (Batch, Time, Freq)
        x = x.permute(0, 2, 1)
        
        # RNN
        x, _ = self.rnn(x)
        
        # FC (sigmoid는 loss function에 포함됨)
        x = self.fc(x)
        
        return x