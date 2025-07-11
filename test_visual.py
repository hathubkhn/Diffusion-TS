import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention_map(attn, H, W, batch_idx=0, head_idx=0, save_path=None):
    """
    Trực quan hóa attention map từ tensor attn.
    
    Args:
        attn (torch.Tensor): Tensor chú ý với kích thước (B * num_heads, H*W, H*W).
        H (int): Chiều cao của đặc trưng không gian.
        W (int): Chiều rộng của đặc trưng không gian.
        batch_idx (int): Chỉ số batch cần trực quan hóa (mặc định: 0).
        head_idx (int): Chỉ số đầu chú ý cần trực quan hóa (mặc định: 0).
        save_path (str): Đường dẫn để lưu heatmap (nếu có).
    """
    num_heads = attn.size(0) // batch_idx if batch_idx > 0 else attn.size(0)
    
    attn_map = attn[batch_idx * num_heads + head_idx]  # (H*W, H*W)
    
    attn_map = attn_map.detach().cpu().numpy()
    
    attn_map = attn_map.reshape(H, W, H, W).mean(axis=(1, 3))  # Trung bình hóa để có (H, W)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_map, cmap='viridis', square=True)
    plt.title(f'Attention Map (Batch {batch_idx}, Head {head_idx})')
    plt.xlabel('Key (W)')
    plt.ylabel('Query (H)')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Giả sử bạn có tensor attn từ CrossAttentionBlock
# Ví dụ: attn có kích thước (B * num_heads, H*W, H*W), với B=1, num_heads=8, H=32, W=32
attn = torch.rand(8, 1024, 1024)  # Ví dụ tensor ngẫu nhiên
H, W = 32, 32  # Kích thước không gian

visualize_attention_map(attn, H, W, batch_idx=0, head_idx=0, save_path='attention_map.png')




                                # # 3. x_future = [mask,future]
                                # hist_part = x_ref[:, :half, :]     # (B, half, top_k)
                                # future_part = x_ref[:, half:, :]   # (B, seq_len - half, top_k)

                                # # Padding
                                # pad_hist = torch.zeros((batch, seq_len - half, top_k), device=x_ref.device, dtype=x_ref.dtype)
                                # pad_future = torch.zeros((batch, half, top_k), device=x_ref.device, dtype=x_ref.dtype)

                                # # Ghép lại để giữ nguyên chiều
                                # x_hist = torch.cat([hist_part, pad_hist], dim=1)    # (B, seq_len, top_k)
                                # x_future = torch.cat([pad_future, future_part], dim=1)  # (B, seq_len, top_k)
    def visualize_attention_map(self, attn, H, W, batch_size, block_idx, epoch, num_epochs, save_dir='attention_maps'):
        """
        Trực quan hóa attention map gốc (H*W, H*W) cho batch 16 và 32, chỉ ở epoch 9.

        Args:
            attn (torch.Tensor): Tensor chú ý với kích thước (B * num_heads, top_k, H*W, H*W) hoặc (B * num_heads, H*W, H*W).
            H (int): Chiều cao của đặc trưng không gian.
            W (int): Chiều rộng của đặc trưng không gian.
            batch_size (int): Kích thước batch.
            block_idx (str): Tên khối để đặt tên file.
            epoch (int): Epoch hiện tại (0-based).
            num_epochs (int): Tổng số epoch.
            save_dir (str): Thư mục lưu heatmap.
        """
        num_epochs = 20
        # if epoch > 0:
            # print(f"Epoch: {epoch}, Num_epochs: {num_epochs}")
        if epoch != (num_epochs - 1):
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        attn = attn.detach().cpu().numpy()  # numpy
        
        # Kiểm tra và xử lý shape của tensor
        if len(attn.shape) == 4:  # (B*heads, top_k, HW, HW)
            # Lấy trung bình theo dimension top_k hoặc chọn top_k đầu tiên
            attn = attn.mean(axis=1)  # hoặc attn = attn[:, 0, :, :]
            print(f"Đã reshape attention từ 4D sang 3D: {attn.shape}")
        elif len(attn.shape) == 3:  # (B*heads, HW, HW)
            pass  # Đã đúng format
        else:
            print(f"Unexpected attention shape: {attn.shape}")
            return
        
        for b in [15, 31]:  # batch 16 và 32 (0-based: 15 và 31)
            if b >= batch_size:  
                continue
            for h in range(self.num_heads):
                idx = b * self.num_heads + h
                if idx >= attn.shape[0]:
                    continue
                    
                attn_map = attn[idx]  # (H*W, H*W)
                
                # Kiểm tra shape của attn_map
                if len(attn_map.shape) != 2:
                    print(f"Skipping invalid attention map shape: {attn_map.shape}")
                    continue
                
                plt.figure(figsize=(10, 10))
                sns.heatmap(attn_map, cmap='viridis', square=True, cbar=True)
                plt.title(f'Attention Map (Epoch {epoch+1}, Batch {b+1}, Head {h}, Block {block_idx})')
                plt.xlabel('Key Positions in ref (H*W)')
                plt.ylabel('Query Positions in x (H*W)')
                
                save_path = os.path.join(save_dir, f'attn_epoch_{epoch+1}_batch_{b+1}_head_{h}_block_{block_idx}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        print("oke đã visual")



# Thay đổi x_hist và x_future
# 1. x_hist/fut đều kích thước 40, mask = 0 
                                # # Tạo tensor toàn 0 cùng shape
                                # x_hist = torch.zeros_like(x_ref)
                                # x_future = torch.zeros_like(x_ref)
                                # # Gán nửa đầu cho x_hist
                                # x_hist[:, :half, :] = x_ref[:, :half, :]
                                # # Gán nửa sau cho x_future
                                # x_future[:, :seq_len - half, :] = x_ref[:, half:, :]