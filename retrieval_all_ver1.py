import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Định nghĩa mô hình TCN Encoder
# -------------------------------
class TCNEncoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, embedding_dim=64):
        super(TCNEncoder, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], embedding_dim)

    def forward(self, x):
        # x shape = (B, input_size, window_length)
        y = self.network(x)          # (B, out_channels, window_length)
        y = torch.mean(y, dim=2)     # Global Average Pooling => (B, out_channels)
        return self.fc(y)            # => (B, embedding_dim)

# -------------------------------
# 2. Hàm tạo sliding windows cho ma trận
# -------------------------------
def create_sliding_windows(matrix, window_size, stride=1):
    """
    matrix shape = (T, num_features)
    Trả về tensor shape (num_windows, window_size, num_features)
      => Mỗi window: ma trận (window_size, num_features)
    """
    data_list = []
    T = len(matrix)
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        data_list.append(matrix[start:end])
    return np.array(data_list)  # shape (num_windows, window_size, num_features)

# -------------------------------
# 3. Main: Tạo file reference_all_stock.pt
# -------------------------------
def main():
    # Tham số
    csv_path = "sp500_industry.csv"
    seq_len = 30  # số ngày lịch sử
    pred_len = 7  # số ngày dự báo
    window_size = seq_len + pred_len  # độ dài mỗi window

    # 1. Đọc CSV, pivot để mỗi cột là 1 cổ phiếu
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Pivot: index=Date, columns=Symbol, values=Close
    pivot_df = df.pivot(index='Date', columns='Symbol', values='Close')
    pivot_df.sort_values(['Date'], inplace=True)  # Đảm bảo theo trật tự thời gian

    # (T, num_stocks)
    matrix = pivot_df.values.astype(np.float32)

    # Xử lý missing nếu cần, ví dụ fillna bằng forward fill (hoặc dropna)
    # pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')
    # matrix = pivot_df.values.astype(np.float32)

    print("Pivot shape:", matrix.shape, "=> (T, num_stocks)")

    # 2. Scale dữ liệu
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)  # shape (T, num_stocks)

    # 3. Tạo sliding windows => shape (num_windows, window_size, num_stocks)
    windows = create_sliding_windows(matrix_scaled, window_size=window_size, stride=1)
    num_windows = windows.shape[0]
    num_stocks = windows.shape[2]
    print("Number of windows:", num_windows)
    print("Number of stocks:", num_stocks)

    # 4. Chuyển windows => tensor cho TCNEncoder
    # TCNEncoder yêu cầu shape (B, input_size, window_length)
    # => windows hiện shape (num_windows, window_size, num_stocks)
    # => ta chuyển thành (num_windows, num_stocks, window_size)
    windows_tensor = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)

    # 5. Khởi tạo TCNEncoder (input_size = num_stocks)
    encoder = TCNEncoder(input_size=num_stocks, num_channels=[16, 32], kernel_size=3, dropout=0.2, embedding_dim=64)
    encoder.eval()

    # 6. Tính embedding cho toàn bộ windows
    with torch.no_grad():
        embeddings = encoder(windows_tensor)  # (num_windows, embedding_dim)
    embeddings_np = embeddings.cpu().numpy()

    # 7. Với mỗi window, tìm top-5 windows gần nhất
    reference_indices = []
    for i in range(num_windows):
        query_emb = embeddings_np[i]  # (64,)
        # Tính khoảng cách L2
        dists = np.linalg.norm(embeddings_np - query_emb, axis=1)  # shape (num_windows,)
        dists[i] = np.inf  # loại trừ chính nó
        # Lấy top 5
        top5 = np.argsort(dists)[:5]
        reference_indices.extend(top5.tolist())

    # 8. Chuyển thành tensor
    reference_tensor = torch.tensor(reference_indices, dtype=torch.long)

    # 9. Clamp để tránh index vượt ngoài
    # => max_index = num_windows - 1 - ...
    # Thường code gốc clamp theo (num_samples - seq_len - pred_len),
    # nhưng do logic "all stock" có T-chuỗi, ta clamp an toàn = num_windows - 1
    max_index = num_windows - 1
    reference_tensor = torch.clamp(reference_tensor, min=0, max=max_index)

    # 10. Lưu file
    torch.save(reference_tensor, "reference_all_stock.pt")
    print("Saved reference_all_stock.pt with shape:", reference_tensor.shape)

if __name__ == "__main__":
    main()
