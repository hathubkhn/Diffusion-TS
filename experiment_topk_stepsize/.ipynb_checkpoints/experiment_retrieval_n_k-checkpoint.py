import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Hỗ trợ cho TCN: Chomp1d và TemporalBlock
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
# TCN Encoder định nghĩa cho dữ liệu của GOOG (1 kênh)
# -------------------------------
class TCNEncoder(nn.Module):
    def __init__(self, input_size=1, num_channels=[16, 32], kernel_size=3, dropout=0.2, emb_dim=64):
        """
        input_size: Số kênh đầu vào (ở đây = 1, vì chỉ dùng giá Close của GOOG).
        num_channels: Danh sách số kênh cho các tầng Conv1d.
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
# Hàm tạo sliding windows cho chuỗi 1 chiều
# -------------------------------
def create_sliding_windows(series, window_size, stride=1):
    """
    Nhận vào:
      - series: mảng numpy 1 chiều chứa chuỗi giá.
      - window_size: độ dài mỗi window (ví dụ: seq_len+pred_len).
      - stride: bước nhảy.
    Trả về:
      - numpy array với shape (num_windows, window_size)
    """
    return np.array([series[i:i+window_size] for i in range(0, len(series)-window_size+1, stride)])

# -------------------------------
# Main: Thử nghiệm với các giá trị top_k và stride, lưu kết quả vào folder experiment_n_k
# -------------------------------
def main():
    # Tham số cố định
    csv_path = "sp500_industry.csv"
    seq_len = 96       # Số ngày lịch sử
    pred_len = 24      # Số ngày dự báo
    window_size = seq_len + pred_len  # Tổng độ dài mỗi window (ví dụ: 120)

    # Danh sách các giá trị cần thử
    top_k_list = [1, 3, 5]
    stride_list = [1,5, 10, 20]

    # Tạo folder lưu kết quả
    output_folder = "experiment_n_k"
    os.makedirs(output_folder, exist_ok=True)

    # 1. Đọc dữ liệu và chỉ lấy dữ liệu của GOOG
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df[df['Symbol'] == 'GOOG']
    df = df[['Date', 'Close']]
    df = df.rename(columns={"Date": "timestamp", "Close": "MT_001"})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    # 2. Lấy chuỗi giá và scale dữ liệu
    prices = df['MT_001'].values.astype(np.float32)
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Lặp qua các giá trị stride và top_k
    for stride in stride_list:
        # Tạo sliding windows với stride tương ứng
        windows = create_sliding_windows(prices_scaled, window_size, stride=stride)
        num_samples = windows.shape[0]
        print(f"Stride = {stride} => Number of windows: {num_samples}")

        # Chuyển windows thành tensor: (B, 1, window_size)
        windows_tensor = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)

        # Khởi tạo TCNEncoder với input_size = 1
        encoder = TCNEncoder(input_size=1, num_channels=[16, 32], kernel_size=3, dropout=0.2, emb_dim=64)
        # Load pretrained encoder dành cho dữ liệu của GOOG
        encoder.load_state_dict(torch.load("tcn_encoder_pretrained_only.pt", map_location="cpu"))
        encoder.eval()

        # Tính embedding cho toàn bộ các window
        with torch.no_grad():
            embeddings = encoder(windows_tensor)  # (num_samples, 64)
        embeddings_np = embeddings.cpu().numpy()

        # Lặp qua các giá trị top_k
        for top_k in top_k_list:
            reference_indices = []
            for i in range(num_samples):
                query_emb = embeddings_np[i]
                dists = np.linalg.norm(embeddings_np - query_emb, axis=1)
                dists[i] = np.inf  # loại trừ mẫu hiện tại
                topk_idx = np.argsort(dists)[:top_k]
                reference_indices.extend(topk_idx.tolist())
            
            # Chuyển danh sách các chỉ số thành tensor và clamp
            reference_tensor = torch.tensor(reference_indices, dtype=torch.long)
            max_index = num_samples - 1
            reference_tensor = torch.clamp(reference_tensor, min=0, max=max_index)
            
            # Tạo tên file theo định dạng: reference_{top_k}_{stride}.pt
            filename = f"reference_{top_k}_{stride}.pt"
            filepath = os.path.join(output_folder, filename)
            torch.save(reference_tensor, filepath)
            print(f"Saved {filename} with shape: {reference_tensor.shape}")

if __name__ == "__main__":
    main()
