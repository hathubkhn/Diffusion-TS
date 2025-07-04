1. tải folder và giải nén: https://drive.google.com/drive/folders/11PXAj0RYei5MyXJVasikmYnEDK6V8awt?usp=share_link
--> đổi tên folder đó thành "data" và xóa cái folder "data" trong github gốc

2. đưa file GOOG.csv vào /home/user11/thongt/ImagenTime/data/short_range/GOOG.csv
(chỉ giữ lại cột "Close")

3. tạo file goog.yaml giống như em tạo ở /home/user11/thongt/ImagenTime/configs/extrapolation/goog.yaml

5. Dòng 101 trong file /home/user11/thongt/ImagenTime/utils/utils_data.py sửa thành:
 ori_data = np.loadtxt('./data/short_range/GOOG.csv', delimiter=",", skiprows=1)

6. sau dongf 106 (task evaluation), bổ sung:
x_ts= x_ts.squeeze(0)
x_ts_sampled = x_ts_sampled.squeeze(0)

7. Dòng 74 của file /home/user11/thongt/ImagenTime/models/img_transformations.py, bổ sung:
signal = signal.unsqueeze(0)

8. run: python run_conditional.py --config ./configs/extrapolation/goog.yaml
