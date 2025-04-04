import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """Cắt bỏ phần padding dư phía sau (dùng cho TCN)."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    """Một block cơ bản cho TCN."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNEncoder(nn.Module):
    """
    TCN Encoder đơn giản để trích xuất embedding cho chuỗi thời gian.
    Giả sử đầu vào shape = (B, num_features, seq_len).
    Đầu ra: vector embedding shape = (B, embedding_dim).
    """
    def __init__(self, num_features=1, num_channels=[16, 16, 32], kernel_size=3, dropout=0.2, emb_dim=64):
        super().__init__()
        layers = []
        in_channels = num_features
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            block = TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            )
            layers.append(block)
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        
        # Projection sang embedding dim
        self.proj = nn.Linear(in_channels, emb_dim)
    
    def forward(self, x):
        """
        x shape = (B, num_features, seq_len).
        Output shape = (B, emb_dim).
        """
        # TCN output shape = (B, out_channels, seq_len) 
        feat = self.network(x)
        # Pool theo time axis -> (B, out_channels)
        # Có thể dùng global average pooling hoặc lấy giá trị cuối, v.v.
        feat = torch.mean(feat, dim=-1)  # [B, out_channels]
        emb = self.proj(feat)           # [B, emb_dim]
        return emb
