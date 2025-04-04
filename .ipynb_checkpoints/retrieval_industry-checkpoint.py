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
        # x: (B, input_size, window_length)
        y = self.network(x)          # (B, out_channels, window_length)
        y = torch.mean(y, dim=2)     # Global Average Pooling => (B, out_channels)
        return self.fc(y)            # => (B, embedding_dim)

# -------------------------------
# 2. Hàm tạo sliding windows cho ma trận
# -------------------------------
def create_sliding_windows(matrix, window_size, stride=1):
    """
    matrix shape = (T, num_features)
    Trả về numpy array shape (num_windows, window_size, num_features)
      => Mỗi window: (window_size, num_features)
    """
    data_list = []
    T = len(matrix)
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        data_list.append(matrix[start:end])
    return np.array(data_list)  # shape (num_windows, window_size, num_features)

# -------------------------------
# 3. Main: Tạo file reference_industry.pt
# -------------------------------
def main():
    # Tham số
    csv_path = "sp500_industry.csv"
    seq_len = 30  # số ngày lịch sử
    pred_len = 7  # số ngày dự báo
    window_size = seq_len + pred_len  # độ dài mỗi window

    # 1. Đọc CSV
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.sort_values(['Date', 'Symbol'], inplace=True)

    # Xác định sector của GOOG
    df_goog = df[df['Symbol'] == 'GOOG']
    if len(df_goog) == 0:
        raise ValueError("Không tìm thấy GOOG trong CSV.")
    goog_sector = df_goog.iloc[0]['Sector']
    print("GOOG sector:", goog_sector)

    # Lọc các cổ phiếu cùng sector
    df_industry = df[df['Sector'] == goog_sector].copy()
    print("Number of rows in this industry:", len(df_industry))

    # Pivot => index=Date, columns=Symbol, values=Close
    pivot_df = df_industry.pivot(index='Date', columns='Symbol', values='Close')
    pivot_df.sort_values(['Date'], inplace=True)
    print("Pivot shape:", pivot_df.shape, "(T, num_stocks)")

    # (T, num_stocks)
    matrix = pivot_df.values.astype(np.float32)

    # Xử lý missing (fillna hoặc dropna) nếu cần
    # pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')
    # matrix = pivot_df.values.astype(np.float32)

    # 2. Scale dữ liệu
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)  # shape (T, num_stocks)

    # 3. Tạo sliding windows => shape (num_windows, window_size, num_stocks)
    windows = create_sliding_windows(matrix_scaled, window_size=window_size, stride=1)
    num_windows = windows.shape[0]
    num_stocks = windows.shape[2]
    print("Number of windows:", num_windows)
    print("Number of stocks in this sector:", num_stocks)

    # 4. Chuyển windows => tensor cho TCNEncoder
    # => (num_windows, num_stocks, window_size)
    windows_tensor = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)

    # 5. Khởi tạo TCNEncoder (input_size = num_stocks)
    encoder = TCNEncoder(input_size=num_stocks, num_channels=[16, 32], kernel_size=3,
                         dropout=0.2, embedding_dim=64)
    encoder.eval()

    # 6. Tính embedding
    with torch.no_grad():
        embeddings = encoder(windows_tensor)  # (num_windows, embedding_dim)
    embeddings_np = embeddings.cpu().numpy()

    # 7. Tìm top-5 window gần nhất cho mỗi window
    reference_indices = []
    for i in range(num_windows):
        query_emb = embeddings_np[i]
        dists = np.linalg.norm(embeddings_np - query_emb, axis=1)  # L2 distance
        dists[i] = np.inf  # loại chính nó
        top5 = np.argsort(dists)[:5]
        reference_indices.extend(top5.tolist())

    # 8. Chuyển thành tensor
    reference_tensor = torch.tensor(reference_indices, dtype=torch.long)

    # 9. Clamp
    max_index = num_windows - 1
    reference_tensor = torch.clamp(reference_tensor, min=0, max=max_index)

    # 10. Lưu file
    torch.save(reference_tensor, "reference_industry.pt")
    print("Saved reference_industry.pt with shape:", reference_tensor.shape)

if __name__ == "__main__":
    main()
