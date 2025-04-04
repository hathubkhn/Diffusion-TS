import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation)
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
    def __init__(self, input_size=1, num_channels=[16, 32], kernel_size=3, dropout=0.2, emb_dim=64):
        super().__init__()
        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation, padding, dropout))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(in_channels, emb_dim)

    def forward(self, x):
        feat = self.network(x)               # (B, C, T)
        feat = torch.mean(feat, dim=2)       # Global average pooling -> (B, C)
        emb = self.proj(feat)                # => (B, emb_dim)
        return emb

def create_sliding_windows(series, window_size, stride=1):
    return np.array([series[i:i+window_size] for i in range(0, len(series)-window_size+1, stride)])

def main():
    csv_path = "sp500_industry.csv"
    seq_len = 96
    pred_len = 24
    window_size = seq_len + pred_len
    top_k = 5

    df = pd.read_csv(csv_path)
    df = df[df['Symbol'] == 'GOOG']
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(by='Date')

    values = df['Close'].values.astype(np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    # Full windows (x^H + x^F)
    full_windows = create_sliding_windows(scaled, window_size, stride=1)  # (N, 120)
    xH_windows = full_windows[:, :seq_len]                                # (N, 96)

    xH_tensor = torch.tensor(xH_windows, dtype=torch.float32).unsqueeze(1)  # (N, 1, 96)

    encoder = TCNEncoder()
    encoder.load_state_dict(torch.load("tcn_encoder_pretrained_only.pt", map_location="cpu"))
    encoder.eval()

    with torch.no_grad():
        xH_embeddings = encoder(xH_tensor).cpu().numpy()  # (N, emb_dim)

    reference_indices = []
    for i in range(len(full_windows)):
        query = xH_embeddings[i]
        dists = np.linalg.norm(xH_embeddings - query, axis=1)
        dists[i] = np.inf
        topk = np.argsort(dists)[:top_k]
        reference_indices.extend(topk.tolist())

    reference_tensor = torch.tensor(reference_indices, dtype=torch.long)
    reference_tensor = torch.clamp(reference_tensor, min=0, max=len(full_windows)-1)

    os.makedirs("retrieval_output", exist_ok=True)
    torch.save(reference_tensor, "retrieval_output/reference_only_96.pt")
    print("✅ Saved to retrieval_output/reference_only_96.pt with shape:", reference_tensor.shape)

if __name__ == "__main__":
    main()
