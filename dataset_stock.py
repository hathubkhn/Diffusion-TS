import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class StockDataset(Dataset):
    def __init__(self, csv_path, flag='train', size=None, data_mode='each_stock', symbol='GOOG', scale=True):
        """
        size: [seq_len, label_len, pred_len]
        data_mode: 'each_stock', 'all_stock', 'industry'
        symbol: tên của cổ phiếu (input luôn là GOOG)
        """
        if size is None:
            raise ValueError("cung cấp giá trị size, ví dụ: [30, 0, 7] (history 30 ngày, pred 7 ngày)")
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_mode = data_mode
        self.symbol = symbol
        self.scale = scale
        self.flag = flag

        # Đọc file CSV, chuyển Date sang datetime, sắp xếp theo ngày
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.sort_values('Date', inplace=True)
        
        # ***Input luôn là của một cổ phiếu (GOOG)***
        df_input = df[df['Symbol'] == symbol]
        df_input = df_input[['Date', 'Close']]

        # Lưu lại cột Date cho input
        self.dates = pd.to_datetime(df_input['Date'], utc=True)

        # Lấy giá đóng cửa của GOOG làm input
        data = df_input['Close'].values.reshape(-1, 1)
        self.raw_data = data.astype(np.float32)
        self.dim = self.raw_data.shape[1]

        # Chia dữ liệu theo tỉ lệ: train 70%, valid 20%, test 10%
        total_len = len(self.raw_data)
        # Tính số sample đảm bảo có đủ seq_len+pred_len
        border1s = [0, int(total_len * 0.7) - self.seq_len, int(total_len * 0.7) + int(total_len * 0.2) - self.seq_len]
        border2s = [int(total_len * 0.7), int(total_len * 0.7) + int(total_len * 0.2), total_len]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[self.flag]
        border1 = border1s[set_type]
        border2 = border2s[set_type]
        self.data = self.raw_data[border1:border2]

        # Scaling: chuẩn hóa theo tập train (sử dụng toàn bộ raw_data cho scaling)
        self.scaler = StandardScaler()
        if self.scale:
            train_data = self.raw_data[0:border2s[0]]
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.data)
        
        # ***Phần Reference***
        # Chỉ load reference từ file, không tính toán lại từ input
        if data_mode == 'each_stock':
            # self.reference = torch.load('experiment_k_n/reference_1_50.pt
            self.reference = torch.load('retrieval_output/reference_only_96.pt')
        elif data_mode == 'all_stock':
            self.reference = torch.load('reference_all_stock.pt')
        elif data_mode == 'industry':
            self.reference = torch.load('reference_industry.pt')
        else:
            raise ValueError("data_mode không hợp lệ. Chọn 'each_stock', 'all_stock' hoặc 'industry'.")
        # Clamp reference để đảm bảo index hợp lệ dựa trên số sample input của GOOG
        self.reference = torch.clamp(self.reference, min=0, max=self.data.shape[0] - self.seq_len - self.pred_len)
        print(self.reference)
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len + self.pred_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Tạo dữ liệu tham chiếu (reference) như mẫu, nếu không có file thật
        reference = np.zeros((3 * self.pred_len, self.dim))
        reference[:self.pred_len] = self.data[self.seq_len:self.seq_len+self.pred_len]
        reference[self.pred_len:2*self.pred_len] = self.data[self.seq_len:self.seq_len+self.pred_len]
        reference[2*self.pred_len:3*self.pred_len] = self.data[self.seq_len:self.seq_len+self.pred_len]
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        # Tạo mốc thời gian (timepoints) dựa trên index (có thể điều chỉnh tùy theo logic)
        timepoints = np.arange(self.seq_len + self.pred_len) * 1.0
        # Danh sách các feature id (số lượng feature = self.dim)
        feature_id = np.arange(self.dim) * 1.0
        
        # Tạo mask: observed_mask là mảng toàn 1
        observed_mask = np.ones_like(seq_x)
        # gt_mask được copy từ observed_mask nhưng phần dự báo (last pred_len) được đặt thành 0
        gt_mask = observed_mask.copy()
        gt_mask[-self.pred_len:] = 0.
        
        sample = {
            'observed_data': seq_x, 
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': timepoints,
            'feature_id': feature_id,
            'reference': reference,
        }
        return sample
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_dataloader(csv_path, data_mode='each_stock', symbol='GOOG', size=None, batch_size=16):
    if size is None:
        size = [30, 0, 7]
    train_dataset = StockDataset(csv_path, flag='train', size=size, data_mode=data_mode, symbol=symbol)
    valid_dataset = StockDataset(csv_path, flag='val', size=size, data_mode=data_mode, symbol=symbol)
    test_dataset = StockDataset(csv_path, flag='test', size=size, data_mode=data_mode, symbol=symbol)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader




# import pickle
# from torch.utils.data import DataLoader, Dataset
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.preprocessing import StandardScaler

# class StockDataset(Dataset):
#     def __init__(self, csv_path, flag='train', size=None, data_mode='each_stock', symbol='GOOG', scale=True):
#         """
#         size: [seq_len, label_len, pred_len]
#         data_mode: 'each_stock', 'all_stock', 'industry'
#         symbol: tên của cổ phiếu cần lọc (dùng khi data_mode=='each_stock' hoặc 'industry')
#         """
#         if size is None:
#             raise ValueError("Bạn cần cung cấp giá trị size, ví dụ: [30, 0, 7] (history 30 ngày, pred 7 ngày)")
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         self.data_mode = data_mode
#         self.symbol = symbol
#         self.scale = scale
#         self.flag = flag

#         # Đọc file CSV và sắp xếp theo ngày
#         df = pd.read_csv(csv_path)
#         # Chuyển đổi cột Date sang datetime với UTC
#         df['Date'] = pd.to_datetime(df['Date'], utc=True)
#         # df.sort_values('Date', inplace=True)
#         df = df[df['Symbol'] == symbol]
#         df = df[['Date', 'Close']]

#         # Lưu lại cột Date
#         self.dates = pd.to_datetime(df['Date'], utc=True)
        
#         # Lấy dữ liệu giá đóng cửa
#         # if data_mode == 'each_stock':
#         data = df['Close'].values.reshape(-1, 1)
#         # else:
#         #     data = df.drop(columns=['Date']).values
#         self.raw_data = data.astype(np.float32)
#         self.dim = self.raw_data.shape[1]

#         # Chia dữ liệu theo tỉ lệ: train 70%, valid 20%, test 10%
#         total_len = len(self.raw_data)
#         num_train = int(total_len * 0.7) - self.pred_len - self.seq_len + 1
#         num_valid = int(total_len * 0.2)
#         border1s = [0, int(total_len * 0.7) - self.seq_len, int(total_len * 0.7) + num_valid - self.seq_len]
#         border2s = [int(total_len * 0.7), int(total_len * 0.7) + num_valid, total_len]
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         set_type = type_map[self.flag]
#         border1 = border1s[set_type]
#         border2 = border2s[set_type]
#         self.data = self.raw_data[border1:border2]

#         # Scaling: chuẩn hóa theo tập train
#         self.scaler = StandardScaler()
#         if self.scale:
#             train_data = self.raw_data[0:border2s[0]]
#             self.scaler.fit(train_data)
#             self.data = self.scaler.transform(self.data)
        
#         # Thiết lập phần "reference" cho retrieval.
#         if data_mode == 'each_stock':
#             self.reference = torch.load('reference_GOOG.pt')
#         if data_mode == 'all_stock':
#             self.reference = torch.load('reference_all_stock.pt')
#         if data_mode == 'industry':
#             self.reference = torch.load('reference_industry.pt')
#         self.reference = torch.clamp(self.reference, min=0, max=self.data.shape[0]-self.seq_len-self.pred_len)

    
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len + self.pred_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
        
#         # Tạo dữ liệu tham chiếu (reference) như mẫu, nếu không có file thật
#         reference = np.zeros((3 * self.pred_len, self.dim))
#         reference[:self.pred_len] = self.data[self.seq_len:self.seq_len+self.pred_len]
#         reference[self.pred_len:2*self.pred_len] = self.data[self.seq_len:self.seq_len+self.pred_len]
#         reference[2*self.pred_len:3*self.pred_len] = self.data[self.seq_len:self.seq_len+self.pred_len]
        
#         seq_x = self.data[s_begin:s_end]
#         seq_y = self.data[r_begin:r_end]
        
#         # Tạo mốc thời gian (timepoints) dựa trên index (có thể điều chỉnh tùy theo logic)
#         timepoints = np.arange(self.seq_len + self.pred_len) * 1.0
#         # Danh sách các feature id (số lượng feature = self.dim)
#         feature_id = np.arange(self.dim) * 1.0
        
#         # Tạo mask: observed_mask là mảng toàn 1
#         observed_mask = np.ones_like(seq_x)
#         # gt_mask được copy từ observed_mask nhưng phần dự báo (last pred_len) được đặt thành 0
#         gt_mask = observed_mask.copy()
#         gt_mask[-self.pred_len:] = 0.
        
#         sample = {
#             'observed_data': seq_x, 
#             'observed_mask': observed_mask,
#             'gt_mask': gt_mask,
#             'timepoints': timepoints,
#             'feature_id': feature_id,
#             'reference': reference,
#         }
#         return sample
    
#     def __len__(self):
#         return len(self.data) - self.seq_len - self.pred_len + 1
    
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# def get_dataloader(csv_path, data_mode='each_stock', symbol='GOOG', size=None, batch_size=8):
#     """
#     Hàm tạo DataLoader cho từng chế độ.
#     size: [seq_len, label_len, pred_len]
#     """
#     if size is None:
#         size = [30, 0, 7]  # Ví dụ: 30 ngày lịch sử, dự báo 7 ngày, label_len=0
#     train_dataset = StockDataset(csv_path, flag='train', size=size, data_mode=data_mode, symbol=symbol)
#     valid_dataset = StockDataset(csv_path, flag='val', size=size, data_mode=data_mode, symbol=symbol)
#     test_dataset = StockDataset(csv_path, flag='test', size=size, data_mode=data_mode, symbol=symbol)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, valid_loader, test_loader
