import pandas as pd

file_path = r"H:\iTransformer\iTransformer\dataset\sp500\GOOG_filtered.csv"

# Đọc file CSV
df = pd.read_csv(file_path)

# Đổi tên cột (nếu có)
df.rename(columns={'Date': 'date', 'DATE': 'date'}, inplace=True)

# Lưu lại file gốc
df.to_csv(file_path, index=False)

print("Đã đổi tên cột thành 'date' và lưu lại file gốc.")
