import torch

# # Đường dẫn đến file tensor .pt
# file_path = '/home/user11/binhnkt/ImagenTime_New_flow_3_Backup/ENCODE_IMAGE_UNET/only/ViT-B-32/GOOG/gasf_gadf_linear_trend/40/3_10.pt'

# # Đọc dữ liệu
# data = torch.load(file_path)

# # Kiểm tra kiểu dữ liệu đọc được
# print(type(data))

# # Nếu là tensor, in shape
# if isinstance(data, torch.Tensor):
#     print("Tensor shape:", data.shape)
#     print(data)

# # Nếu là dict, lặp qua các key
# elif isinstance(data, dict):
#     for key, value in data.items():
#         print(f"{key}: type={type(value)}")
#         if isinstance(value, torch.Tensor):
#             print(f"    shape: {value.shape}")


x_ref = torch.tensor([[[1], [2], [3], [4], [5], [6]]], dtype=torch.float)  # (1,6,1)
batch, seq_len, top_k = x_ref.shape
# x_hist = torch.zeros_like(x_ref)
# x_future = torch.zeros_like(x_ref)

half = x_ref.shape[1] // 2

# x_hist = x_ref[:, :half, :]         # (batch, half, top_k)
# x_future = x_ref[:, half:, :]       # (batch, seq_len - half, top_k)
# Nửa đầu và nửa sau
hist_part = x_ref[:, :half, :]     # (B, half, top_k)
future_part = x_ref[:, half:, :]   # (B, seq_len - half, top_k)

# Padding
pad_hist = torch.zeros((batch, seq_len - half, top_k), device=x_ref.device, dtype=x_ref.dtype)
pad_future = torch.zeros((batch, half, top_k), device=x_ref.device, dtype=x_ref.dtype)

# Ghép lại để giữ nguyên chiều
x_hist = torch.cat([hist_part, pad_hist], dim=1)    # (B, seq_len, top_k)
x_future = torch.cat([pad_future, future_part], dim=1)  # (B, seq_len, top_k)
print("x_hist:", x_hist.squeeze().tolist())    # [1,2,3,0,0,0]
print("x_future:", x_future.squeeze().tolist())  # [4,5,6,0,0,0]
