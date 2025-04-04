import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# TCN Encoder hỗ trợ (sử dụng Chomp1d và TemporalBlock)
# -------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[..., :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
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

# -------------------------------
# TCN Encoder định nghĩa theo RATD (cho dữ liệu GOOG)
# -------------------------------
class TCNEncoder(nn.Module):
    def __init__(self, input_size=1, num_channels=[16, 32], kernel_size=3, dropout=0.2, emb_dim=64):
        """
        input_size: Số kênh đầu vào (ở đây = 1 vì chỉ dùng dữ liệu của GOOG).
        num_channels: Danh sách số kênh của các lớp Conv1d, ví dụ [16, 32].
        emb_dim: Chiều của vector embedding cuối cùng.
        """
        super(TCNEncoder, self).__init__()
        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            block = TemporalBlock(in_channels, out_channels, kernel_size,
                                    stride=1, dilation=dilation, padding=padding, dropout=dropout)
            layers.append(block)
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(in_channels, emb_dim)
    
    def forward(self, x):
        # x shape: (B, 1, seq_len)
        feat = self.network(x)         # (B, out_channels, seq_len)
        feat = torch.mean(feat, dim=2)   # Global average pooling -> (B, out_channels)
        emb = self.proj(feat)           # (B, emb_dim)
        return emb

# -------------------------------
# Decoder: hồi phục lại chuỗi đầu vào từ embedding
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, emb_dim, window_size):
        super().__init__()
        self.fc = nn.Linear(emb_dim, window_size)
    
    def forward(self, x):
        # x: (B, emb_dim)
        out = self.fc(x)  # (B, window_size)
        return out.unsqueeze(1)  # (B, 1, window_size)

# -------------------------------
# Autoencoder: kết hợp Encoder và Decoder
# -------------------------------
class TCN_Autoencoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, emb_dim, window_size):
        super(TCN_Autoencoder, self).__init__()
        self.encoder = TCNEncoder(input_size, num_channels, kernel_size, dropout, emb_dim)
        self.decoder = Decoder(emb_dim, window_size)
    
    def forward(self, x):
        emb = self.encoder(x)      # (B, emb_dim)
        recon = self.decoder(emb)  # (B, 1, window_size)
        return recon

# -------------------------------
# Dataset cho Pre-training với dữ liệu của GOOG
# -------------------------------
class GoDataTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, symbol="GOOG", window_size=120, scale=True):
        """
        csv_path: đường dẫn tới file CSV chứa các trường Symbol, Date, Close, Sector.
        symbol: dùng dữ liệu của một cổ phiếu (ví dụ: GOOG).
        window_size: độ dài của mỗi cửa sổ (ví dụ: 120 = seq_len + pred_len).
        scale: chuẩn hóa dữ liệu.
        """
        df = pd.read_csv(csv_path)
        # Lọc chỉ dữ liệu của cổ phiếu được chỉ định (GOOG)
        df = df[df['Symbol'] == symbol]
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values('Date')
        # Lấy cột Close
        prices = df['Close'].values.astype(np.float32)
        self.window_size = window_size
        self.scale = scale
        self.scaler = StandardScaler()
        if self.scale:
            prices = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        self.prices = prices
        # Tạo sliding windows
        self.windows = np.array([prices[i:i+window_size] for i in range(0, len(prices)-window_size+1)])
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        window = self.windows[index]  # shape (window_size,)
        # Chuyển thành tensor với shape (1, window_size) (input_size = 1)
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0)

# -------------------------------
# Pre-train Encoder với dữ liệu của GOOG và lưu checkpoint
# -------------------------------
def pretrain_encoder_only(csv_path, symbol="GOOG", seq_len=96, pred_len=24, batch_size=32, epochs=50, lr=1e-3, device="cuda:0"):
    window_size = seq_len + pred_len  # ví dụ: 120
    dataset = GoDataTimeSeriesDataset(csv_path, symbol=symbol, window_size=window_size, scale=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Với dữ liệu của GOOG, input_size = 1
    model = TCN_Autoencoder(
        input_size=1,
        num_channels=[16, 32],
        kernel_size=3,
        dropout=0.2,
        emb_dim=64,
        window_size=window_size
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)  # (B, 1, window_size)
            optimizer.zero_grad()
            recon = model(batch)      # (B, 1, window_size)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Lưu encoder đã được pre-trained vào file "tcn_encoder_pretrained_only.pt"
    torch.save(model.encoder.state_dict(), "tcn_encoder_pretrained_only.pt")
    print("Pretrained encoder saved to tcn_encoder_pretrained_only.pt")

if __name__ == "__main__":
    csv_path = "sp500_industry.csv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pretrain_encoder_only(csv_path, symbol="GOOG", seq_len=96, pred_len=24, batch_size=32, epochs=50, lr=1e-3, device=device)
