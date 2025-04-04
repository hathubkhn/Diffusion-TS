import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import RATD_Forecasting
from dataset_stock import get_dataloader
# from dataset_stock_n_k import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="RATD for Stock Forecasting")
parser.add_argument("--config", type=str, default="base_stock.yaml")
parser.add_argument("--data_mode", type=str, default="each_stock", choices=["each_stock", "all_stock", "industry"],
                    help="Chế độ dữ liệu: each_stock, all_stock, hoặc industry")
parser.add_argument("--symbol", type=str, default="GOOG", help="Chỉ dùng khi data_mode=='each_stock'")
parser.add_argument("--csv_path", type=str, default="sp500_industry.csv")
parser.add_argument("--device", default="cuda:0", help="Thiết bị chạy model")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=50)
parser.add_argument("--target_dim", type=int, default=1, help="Số lượng feature dự báo; mỗi stock chỉ có 1 chiều, còn all_stock hay industry sẽ có nhiều chiều hơn")
parser.add_argument("--h_size", type=int, default=96, help="Số ngày lịch sử (seq_len)")
parser.add_argument("--ref_size", type=int, default=24, help="Số ngày dự báo (pred_len)")





args = parser.parse_args()
print(args)

# Đọc file cấu hình
with open("config/" + args.config, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["diffusion"]["h_size"] = args.h_size
config["diffusion"]["ref_size"] = args.ref_size
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_stock_" + args.data_mode + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


# reference_file = args.reference_file if args.reference_file != "" else None

# Lấy DataLoader từ dataset_stock.py
train_loader, valid_loader, test_loader = get_dataloader(
    args.csv_path, 
    data_mode=args.data_mode, 
    symbol=args.symbol, 
    size=[args.h_size, 0, args.ref_size], 
    batch_size=config["train"]["batch_size"]
)

model = RATD_Forecasting(config, args.device, args.target_dim).to(args.device)

if args.modelfolder == "":
    train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername)
else:
    model.load_state_dict(torch.load("/save/" + args.modelfolder + "/model.pth"))
model.target_dim = args.target_dim
evaluate(model, test_loader, nsample=args.nsample, scaler=1, mean_scaler=0, foldername=foldername)




# import matplotlib.pyplot as plt
# import torch
# import numpy as np

# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# def plot_3lines_bridged(
#     history_vals,      # shape (30,)  => chỉ số 0..29
#     ground_truth_vals, # shape (7,)   => chỉ số 30..36
#     prediction_vals,   # shape (7,)   => chỉ số 30..36
#     save_path=None
# ):
#     """
#     Vẽ 3 đường:
#       - History (x=0..29)
#       - Ground Truth (x=30..36), liền mạch với History
#       - Prediction (x=30..36), nhưng cũng nối liền với ngày 30 (x=29).
#     """

#     seq_len = len(history_vals)      # 30
#     pred_len = len(ground_truth_vals)# 7

#     # -------------------------
#     # 1) Trục x cho History
#     # -------------------------
#     x_history = np.arange(seq_len)  # 0..29

#     # -------------------------
#     # 2) Trục x cho ground truth
#     # => muốn nối liền? Thêm x=seq_len-1 (tức 29) để ghép
#     # -------------------------
#     x_gt = np.arange(seq_len-1, seq_len + pred_len)  # 29..36
#     # => y_gt = [history_vals[-1], ground_truth_vals...]
#     y_gt = np.concatenate([[history_vals[-1]], ground_truth_vals])

#     # -------------------------
#     # 3) Trục x cho prediction
#     # => cũng nối liền day 30 => Thêm x=29
#     # -------------------------
#     x_pred = np.arange(seq_len-1, seq_len + pred_len)  # 29..36
#     # => y_pred = [history_vals[-1], prediction_vals...]
#     y_pred = np.concatenate([[history_vals[-1]], prediction_vals])

#     plt.figure(figsize=(8,5))

#     # Vẽ History (màu xanh)
#     plt.plot(x_history, history_vals, label="History", color="blue")

#     # Vẽ Ground Truth (màu đen)
#     plt.plot(x_gt, y_gt, label="Ground Truth", color="black")

#     # Vẽ Prediction (màu đỏ)
#     plt.plot(x_pred, y_pred, label="Prediction", color="red", linestyle="--")

#     plt.xlabel("Time Steps (day index)")
#     plt.ylabel("Price")
#     plt.title("Forecast Visualization (Bridged at day 30)")
#     plt.legend()
#     plt.grid(True)

#     if save_path:
#         plt.savefig(save_path)
#         print(f"Plot saved to {save_path}")
#     plt.show()

# def example_plot_bridged(samples, observed_data, sample_idx=0, seq_len=args.h_size, pred_len=args.ref_size, save_path=None):
#     """
#     Giả sử:
#       - samples.shape = (B, nsample, L, K), L=seq_len+pred_len=37
#       - observed_data.shape = (B, L, K)
#     Indexing:
#       History = [0..29], Future = [30..36]
#     """

#     # Lấy sample
#     forecast_samples = samples[sample_idx]  # (nsample, 37, K)
#     obs_data = observed_data[sample_idx]    # (37, K)

#     # Tính median
#     median_forecast = torch.median(forecast_samples, dim=0)[0]  # (37, K)

#     # Tách history, ground_truth, prediction
#     history_vals = obs_data[:seq_len, 0].cpu().numpy()                    # (30,)
#     ground_truth_vals = obs_data[seq_len:seq_len+pred_len, 0].cpu().numpy()  # (7,)
#     prediction_vals = median_forecast[seq_len:seq_len+pred_len, 0].cpu().numpy() # (7,)

#     # Gọi hàm plot
#     plot_3lines_bridged(
#         history_vals,
#         ground_truth_vals,
#         prediction_vals,
#         save_path=save_path
#     )



# results = evaluate(model, test_loader, nsample=args.nsample, scaler=1, mean_scaler=0, foldername=foldername)

# if results is not None:
#     all_generated_samples, all_target, all_evalpoint, all_observed_point, all_observed_time = results
    
#     # Vẽ cho sample đầu tiên:
#     sample_idx = 0
#     example_plot_bridged(
#     samples=all_generated_samples,
#     observed_data=all_target,
#     sample_idx=0,
#     seq_len=args.h_size,
#     pred_len=args.ref_size,
#     save_path="forecast_bridged.png"
# )


# else:
#     print("No results returned from evaluate().")
# example_plot_bridged(
#     samples=all_generated_samples,
#     observed_data=all_target,
#     sample_idx=0,
#     seq_len=args.h_size,
#     pred_len=args.ref_size,
#     save_path="forecast_bridged.png"
# )


