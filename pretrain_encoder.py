import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Các lớp hỗ trợ cho TCN: Chomp1d & TemporalBlock
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
# TCN Encoder
# -------------------------------
class TCNEncoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, emb_dim=64):
        """
        input_size: số kênh đầu vào = số cổ phiếu (num_stocks)
        num_channels: danh sách số kênh cho các tầng Conv1d (ví dụ: [16, 32])
        emb_dim: chiều vector embedding cuối cùng
        """
        super(TCNEncoder, self).__init__()
        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            block = TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                    dilation=dilation_size, padding=padding, dropout=dropout)
            layers.append(block)
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(in_channels, emb_dim)
    
    def forward(self, x):
        # x: (B, input_size, seq_len)
        feat = self.network(x)         # (B, out_channels, seq_len)
        feat = torch.mean(feat, dim=2)   # Global average pooling -> (B, out_channels)
        emb = self.proj(feat)           # (B, emb_dim)
        return emb

# -------------------------------
# Decoder: Dự đoán lại toàn bộ window
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, emb_dim, window_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(emb_dim, window_size)
    
    def forward(self, x):
        # x: (B, emb_dim)
        out = self.fc(x)  # (B, window_size)
        return out.unsqueeze(1)  # (B, 1, window_size)

# -------------------------------
# Autoencoder: Encoder + Decoder
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
# Dataset cho Pre-training toàn bộ dữ liệu time series
# -------------------------------
class AllStockTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, scale=True, window_size=120):
        """
        Sử dụng toàn bộ dữ liệu time series của SP500.
        Dữ liệu được pivot để mỗi hàng là 1 ngày và mỗi cột là giá đóng cửa của 1 cổ phiếu.
        window_size: độ dài của mỗi cửa sổ (ví dụ: 120 = seq_len + pred_len)
        """
        # Đọc CSV (giả sử dữ liệu đã được sắp xếp theo Date)
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        # Pivot dữ liệu: index=Date, columns=Symbol, values=Close
        pivot_df = df.pivot(index='Date', columns='Symbol', values='Close')
        # Giả sử dữ liệu đã sắp xếp theo date, không cần sort lại
        self.data = pivot_df.values.astype(np.float32)  # shape (T, num_stocks)
        self.window_size = window_size
        self.scale = scale
        self.scaler = StandardScaler()
        if self.scale:
            self.data = self.scaler.fit_transform(self.data)
        
        # Tạo sliding windows theo chiều thời gian
        T = self.data.shape[0]
        windows = []
        for i in range(0, T - window_size + 1):
            windows.append(self.data[i:i+window_size])
        self.windows = np.array(windows)  # shape (num_windows, window_size, num_stocks)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        # Trả về window với shape (num_stocks, window_size)
        window = self.windows[index]  # (window_size, num_stocks)
        return torch.tensor(window, dtype=torch.float32).permute(1, 0)  # (num_stocks, window_size)

# -------------------------------
# Pre-train Encoder qua Autoencoder Reconstruction trên toàn bộ dữ liệu
# -------------------------------
def pretrain_encoder_all(csv_path, window_size=120, batch_size=32, epochs=50, lr=1e-3, device="cuda:0"):
    dataset = AllStockTimeSeriesDataset(csv_path, scale=True, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # input_size = số cổ phiếu (số cột sau pivot)
    num_stocks = dataset.data.shape[1]
    model = TCN_Autoencoder(
        input_size=num_stocks,
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
            batch = batch.to(device)  # (B, num_stocks, window_size)
            optimizer.zero_grad()
            recon = model(batch)      # (B, 1, window_size)
            loss = criterion(recon, batch.unsqueeze(1))  # so sánh (B, 1, window_size)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Lưu encoder đã được pre-trained
    torch.save(model.encoder.state_dict(), "tcn_encoder_pretrained_all.pt")
    print("Pretrained encoder saved to tcn_encoder_pretrained_all.pt")

if __name__ == "__main__":
    csv_path = "sp500_industry.csv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pretrain_encoder_all(csv_path, window_size=120, batch_size=32, epochs=50, lr=1e-3, device=device)
