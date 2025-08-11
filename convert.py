import pandas as pd
import ast  # để chuyển từ chuỗi sang dictionary

# Bước 1: Đọc file CSV
df = pd.read_csv('result_60_60.csv')

# Bước 2: Chuyển chuỗi thành dict và lấy giá trị
df['Best_MSE'] = df['Best_MSE'].apply(lambda x: list(ast.literal_eval(x).values())[0])
df['Best_MAE'] = df['Best_MAE'].apply(lambda x: list(ast.literal_eval(x).values())[0])
# Bước 3: Ghi lại file nếu muốn
df.to_csv('results_60_60.csv', index=False)

print(df)


