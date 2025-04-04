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
# TCN Encoder định nghĩa theo RATD (trích xuất embedding)
# -------------------------------
class TCNEncoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, emb_dim=64):
        """
        input_size: Số kênh đầu vào. Ở đây, khi dùng toàn bộ dữ liệu time series sau pivot,
                    input_size = số cổ phiếu (num_stocks).
        num_channels: Danh sách số kênh cho các tầng Conv1d (ví dụ: [16, 32]).
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
        # x shape: (B, input_size, seq_len)
        feat = self.network(x)         # (B, out_channels, seq_len)
        feat = torch.mean(feat, dim=2)   # Global average pooling -> (B, out_channels)
        emb = self.proj(feat)           # (B, emb_dim)
        return emb

# -------------------------------
# Hàm tạo sliding windows
# -------------------------------
def create_sliding_windows(matrix, window_size, stride=1):
    """
    Nhận vào:
      - matrix: mảng numpy có shape (T, num_features) 
      - window_size: độ dài mỗi window (ví dụ: seq_len+pred_len)
      - stride: bước nhảy (mặc định = 1)
    Trả về:
      - numpy array có shape (num_windows, window_size, num_features)
    """
    return np.array([matrix[i:i+window_size] for i in range(0, len(matrix)-window_size+1, stride)])

# -------------------------------
# Main: Tạo file reference_all_stock.pt
# -------------------------------
def main():
    # Tham số
    csv_path = "sp500_industry.csv"
    seq_len = 96   # Số ngày lịch sử
    pred_len = 24  # Số ngày dự báo
    window_size = seq_len + pred_len  # Tổng độ dài mỗi window, ví dụ 120
    top_k = 5  # Số mẫu reference cần lấy

    # 1. Đọc dữ liệu từ CSV và pivot dữ liệu
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    # Dữ liệu đã được sắp xếp theo Date, không cần sắp xếp lại
    pivot_df = df.pivot(index='Date', columns='Symbol', values='Close')
    # pivot_df có shape (T, num_stocks)
    data = pivot_df.values.astype(np.float32)

    # 2. Scale dữ liệu
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 3. Tạo sliding windows theo chiều thời gian: mỗi window có shape (window_size, num_stocks)
    windows = create_sliding_windows(data_scaled, window_size, stride=1)
    num_windows = windows.shape[0]
    print("Number of windows:", num_windows)

    # 4. Chuyển windows thành tensor: mong đợi đầu vào cho TCNEncoder có shape (B, input_size, seq_len)
    # Ở đây, input_size = số cổ phiếu, seq_len = window_size
    windows_tensor = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)  # (num_windows, num_stocks, window_size)

    # 5. Khởi tạo TCNEncoder với input_size = num_stocks
    num_stocks = data_scaled.shape[1]
    encoder = TCNEncoder(input_size=num_stocks, num_channels=[16, 32], kernel_size=3, dropout=0.2, emb_dim=64)
    # Load pretrained encoder từ checkpoint đã lưu (tcn_encoder_pretrained_all.pt)
    encoder.load_state_dict(torch.load("tcn_encoder_pretrained_all.pt", map_location="cpu"))
    encoder.eval()

    # 6. Tính embedding cho toàn bộ các window
    with torch.no_grad():
        embeddings = encoder(windows_tensor)  # (num_windows, 64)
    embeddings_np = embeddings.cpu().numpy()

    # 7. Với mỗi window, tìm top_k nearest windows (trừ chính nó) dựa trên khoảng cách L2
    reference_indices = []
    for i in range(num_windows):
        query_emb = embeddings_np[i]
        # Tính khoảng cách L2 giữa query_emb và tất cả embeddings
        dists = np.linalg.norm(embeddings_np - query_emb, axis=1)
        dists[i] = np.inf  # loại trừ mẫu hiện tại
        topk_idx = np.argsort(dists)[:top_k]
        reference_indices.extend(topk_idx.tolist())

    # 8. Chuyển danh sách các chỉ số thành tensor và clamp
    reference_tensor = torch.tensor(reference_indices, dtype=torch.long)
    max_index = num_windows - 1
    reference_tensor = torch.clamp(reference_tensor, min=0, max=max_index)

    # 9. Lưu tensor reference vào file (ví dụ: reference_all_stock.pt)
    torch.save(reference_tensor, "reference_all_stock.pt")
    print("Saved reference_all_stock.pt with shape:", reference_tensor.shape)

if __name__ == "__main__":
    main()
