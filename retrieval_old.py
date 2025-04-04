import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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
# Hàm tạo sliding windows
# -------------------------------
def create_sliding_windows(series, window_size, stride=1):
    """
    Trả về mảng numpy có shape (num_windows, window_size)
    """
    return np.array([series[i:i+window_size] for i in range(0, len(series)-window_size+1, stride)])

# -------------------------------
# Tạo file reference_GOOG.pt
# -------------------------------
def main():
    # Thiết lập tham số 
    csv_path = "sp500_industry.csv"
    seq_len = 96  # Số ngày lịch sử
    pred_len = 24 # Số ngày dự báo
    window_size = seq_len + pred_len  # Mỗi mẫu là 1 cửa sổ có độ dài này

    # 1. Đọc dữ liệu và lọc GOOG
    df = pd.read_csv(csv_path)
    df = df[df['Symbol'] == 'GOOG']
    df = df[['Date', 'Close']]
    df = df.rename(columns={"Date": "timestamp", "Close": "MT_001"})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    # 2. Lấy chuỗi giá và scale dữ liệu
    prices = df['MT_001'].values.astype(np.float32)
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1,1)).flatten()

    # 3. Tạo sliding windows (mỗi window có độ dài = seq_len + pred_len)
    windows = create_sliding_windows(prices_scaled, window_size, stride=1)
    num_samples = windows.shape[0]
    print("Number of samples:", num_samples)

    # 4. Chuyển windows thành tensor cho TCNEncoder
    # Mỗi window có shape (window_size,) -> reshape thành (1, 1, window_size)
    windows_tensor = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)  # (num_samples, 1, window_size)

    # 5. Khởi tạo TCNEncoder
    encoder = TCNEncoder(input_size=1, num_channels=[16, 32], kernel_size=3, dropout=0.2, embedding_dim=64)
    encoder.eval()  # Ở bước này, ta dùng encoder chưa được huấn luyện 

    # 6. Tính embedding cho toàn bộ các mẫu
    with torch.no_grad():
        embeddings = encoder(windows_tensor)  # (num_samples, 64)
    embeddings_np = embeddings.cpu().numpy()

    # 7. Với mỗi mẫu, tìm 5 mẫu (indices) có embedding gần nhất (loại trừ chính mẫu đó)
    reference_indices = []
    for i in range(num_samples):
        query_emb = embeddings_np[i]  # (64,)
        # Tính khoảng cách L2 giữa query_emb và tất cả các embedding
        dists = np.linalg.norm(embeddings_np - query_emb, axis=1)  # (num_samples,)
        # Loại trừ mẫu hiện tại
        dists[i] = np.inf
        # Lấy top 5 chỉ số có khoảng cách nhỏ nhất
        top3 = np.argsort(dists)[:5]
        reference_indices.extend(top3.tolist())

    # 8. Chuyển reference_indices thành tensor và clamp
    reference_tensor = torch.tensor(reference_indices, dtype=torch.long)
    max_index = num_samples - seq_len - pred_len  # đảm bảo index không vượt quá phạm vi
    reference_tensor = torch.clamp(reference_tensor, min=0, max=max_index)

    # 9. Lưu tensor này vào file reference_GOOG.pt
    torch.save(reference_tensor, "reference_GOOG_96.pt")
    print("Saved reference_GOOG.pt with shape:", reference_tensor.shape)

if __name__ == "__main__":
    main()
