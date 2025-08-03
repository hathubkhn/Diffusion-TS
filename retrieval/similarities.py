import torch
import numpy as np
import pandas as pd
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
input_file = f'/home/user11/thongt/Diffusion-TS-all-stock/data_new/GOOG_only_1.csv'
df = pd.read_csv(input_file, usecols=['Date', 'Symbol', 'Close'])

query = torch.tensor(df['Close'][:40].values, dtype=torch.float32).unsqueeze(0)
ref_file = '/home/user11/thongt/ImagenTime_New_flow_3_Backup_2_shuffle/Database/GOOG/VGG16/4/only_gasf_gadf/20.pt'
ref = torch.load(ref_file)
ref_3 = ref[:120]
ref_3 = ref_3.reshape(3, 40)
print(query)
print(ref_3)
#Normalize
query_repeat = query.repeat(3, 1)
# Tính similarity
similarities = cosine_similarity(query_repeat, ref_3)
print(similarities)

# tensor([0.9992, 0.9992, 0.9984]) ==> chứng tỏ nếu dùng cosine similarities cho chuỗi (thay vì ảnh) thì sẽ không work.
# Các giá trị gần nhau (40 giá trị) sẽ tương đối giống nhau, VD thuộc khoảng (500, 510)
# Các sample test cũng tương tự, ví dụ (900, 910) ==> query và similarity tỷ lệ với nhau ==> similarities luôn bằng 1

