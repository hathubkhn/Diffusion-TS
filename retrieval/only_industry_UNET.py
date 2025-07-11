import argparse
import sys
from PIL import Image
import pandas as pd
import torch
import open_clip
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Image embedding and retrieval for stock symbols")
    parser.add_argument('--symbols', nargs='+', default=["GOOG", "AMZN", "AAPL"],
                        help='Danh sách các stock symbols cần xử lý (VD: GOOG AMZN AAPL)')
    parser.add_argument('--convert_method', type=str, default='gasf_gadf', nargs='+',
                        choices=['gasf_gadf', 'gasf_gadf_difference', 'gasf_gadf_linear_trend'],
                        help='Phương pháp chuyển đổi ảnh (VD: gasf_gadf, gasf_gadf_difference, gasf_gadf_linear_trend)')
    # parser.add_argument('--dir_path', type=str, default="only_gasf_gadf",
    #                     help='Thư mục chứa ảnh của mỗi stock')
    parser.add_argument('--top_k_list', nargs='+', type=int, default=[1, 2, 3, 5, 10, 20],
                        help='Danh sách các giá trị top-k dùng để truy xuất')
    parser.add_argument('--step_sizes', nargs='+', type=int, default=[1, 2, 5, 10],
                        help='Danh sách các giá trị bước (step_size) dùng để truy xuất')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size cho encode ảnh')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Chiều dài chuỗi đầu vào (sequence length)')
    parser.add_argument('--pred_len', type=int, default=20,
                        help='Chiều dài chuỗi dự đoán (prediction length)')
    parser.add_argument('--model_name', type=str, default='ViT-B-32',
                        choices=['ViT-B-32', 'ViT-L-14', 'ViT-H-14'],
                        help='Tên mô hình OpenCLIP để sử dụng')
    parser.add_argument('--run_type', type=str, default='only',
                        choices=['only', 'industry', 'all'],
                        help='Chạy mô hình chỉ với ảnh hay kết hợp với dữ liệu ngành hoặc all')
    return parser.parse_args()

def load_and_preprocess_image(file_path):
    try:
        img = Image.open(file_path).convert("RGB")
        return img
    except Exception as e:
        print(f"Lỗi khi đọc ảnh {file_path}: {e}")
        return None

def load_images_in_order(image_dir, date_keys, max_workers=64):
    file_paths = [os.path.join(image_dir, f"{key}.png") for key in date_keys]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tensors = list(executor.map(load_and_preprocess_image, file_paths))
    return [t for t in tensors if t is not None]

def main():
    args = parse_args()
    convert_method = args.convert_method
    top_k_list = args.top_k_list
    step_sizes = args.step_sizes
    all_query_symbol = args.symbols
    BATCH_SIZE = args.batch_size
    seq_len = args.seq_len
    pred_len = args.pred_len
    window_size = seq_len + pred_len
    run_type = args.run_type
    model_name = args.model_name


    symbol_to_date_key = defaultdict(list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("DEBUG model_name:", model_name, type(model_name))

    if model_name == 'ViT-B-32':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )
    elif model_name == 'ViT-L-14':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='dfn2b', device=device
        )
    elif model_name == 'ViT-H-14':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14-378-quickgelu', pretrained='dfn5b', device=device
        )
    else:
        raise ValueError(f"Không hỗ trợ model_name: {model_name}")

    

    model.eval()
    
    for query_symbol in all_query_symbol:
        for convert_method in args.convert_method:
            dir_path = f"{run_type}_{convert_method}"
            print("DEBUG run_type:", run_type, type(run_type))
            print(f"\n--- Xử lý symbol: {query_symbol} ---")
            
            if 'all' in run_type:
                dir_path = f"{run_type}_{convert_method}"
                csv_path = f"/home/user11/thongt/Diffusion-TS-all-stock/data_new/all_1.csv"
                image_dir = f"/home/user11/thongt/ImagenTime_New_flow_3_Backup/TS2IMG_UNET/GOOG/GOOG_all_gasf_gadf_linear_trend/{seq_len}" # if database is all, the image folder is the same for all symbols
            else:
                csv_path = f"/home/user11/thongt/Diffusion-TS-all-stock/data_new/{query_symbol}_{run_type}_1.csv"
                image_dir = f"/home/user11/thongt/ImagenTime_New_flow_3_Backup/TS2IMG_UNET/{query_symbol}/{query_symbol}_{dir_path}/{seq_len}"
            
            query_image_tensors = []
            reference_image_tensors = []
            query_date_keys = []
            reference_date_keys = []
            reference_symbols = []

            df = pd.read_csv(csv_path, usecols=['Date', 'Symbol', 'Close'])
            df.columns = df.columns.str.strip()
            df['Date'] = pd.to_datetime(df['Date'])

            for symbol in tqdm(df['Symbol'].unique(), desc="Xử lý symbol"): #df['Symbol'].unique()
                df_symbol = df[df['Symbol'] == symbol]
                if len(df_symbol) < window_size:
                    continue

                df_symbol = df_symbol.sort_values('Date')
                values = df_symbol['Close'].values.astype(np.float32)
                scaler = StandardScaler()
                scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
                dates = df_symbol['Date'].values

                date_keys = [
                    f"{symbol}_{pd.to_datetime(dates[i + pred_len - 1]).strftime('%Y%m%d')}"
                    for i in range(len(scaled) - window_size + 1)
                ]

                print("Đang load ảnh...")
                images = load_images_in_order(image_dir, date_keys)
                if len(images) != len(date_keys):
                    print(f"Bỏ qua {symbol}, thiếu ảnh")
                    continue

                processed_tensors = []

                for i in range(0, len(images), BATCH_SIZE):
                    batch_imgs = images[i:i + BATCH_SIZE]
                    batch_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device, non_blocking=True)

                    with torch.no_grad():
                        batch_embeddings = model.encode_image(batch_tensors)
                        batch_embeddings = batch_embeddings.cpu()
                        processed_tensors.append(batch_embeddings)

                embeddings = torch.cat(processed_tensors)
                # print(symbol)
                # print(query_symbol)
                num_samples = len(date_keys)
                print(num_samples)
                train_cutoff = int(0.7 * num_samples)
                print(train_cutoff)
                if symbol == query_symbol:
                    for i, date_key in enumerate(date_keys):
                        # if symbol == query_symbol:                       
                        query_image_tensors.append(embeddings[i])
                        query_date_keys.append(date_key)
                        if i < train_cutoff:
                            reference_image_tensors.append(embeddings[i])
                            reference_date_keys.append(date_key)
                            reference_symbols.append(symbol)
                else:
                    for i, date_key in enumerate(date_keys):
                        reference_image_tensors.append(embeddings[i])
                        reference_date_keys.append(date_key)
                        reference_symbols.append(symbol)

            query_tensor = torch.stack(query_image_tensors)
            reference_tensor = torch.stack(reference_image_tensors)

            with torch.no_grad():
                query_embeddings = query_tensor / query_tensor.norm(dim=-1, keepdim=True)
                reference_embeddings = reference_tensor / reference_tensor.norm(dim=-1, keepdim=True)

            reference_index_map = {}
            for idx, (symbol, date_key) in enumerate(zip(reference_symbols, reference_date_keys)):
                reference_index_map[idx] = {"symbol": symbol, "date_key": date_key}

            print(f"len(reference_symbols): {len(reference_symbols)}, len(reference_embeddings): {len(reference_embeddings)}")

            client = QdrantClient(":memory:")
            client.create_collection(
                collection_name="reference_vectors",
                vectors_config=VectorParams(size=reference_embeddings.shape[1], distance=Distance.COSINE)
            )

            reference_points = [
                PointStruct(
                    id=int(i),
                    vector=vec.tolist(),
                    payload={
                        "symbol": reference_symbols[i],
                        "date_key": reference_date_keys[i]
                    }
                )
                for i, vec in enumerate(reference_embeddings)
            ]

            client.upsert(collection_name="reference_vectors", points=reference_points)

            for step_size in step_sizes:
                print(f"\n--- Step size = {step_size} ---")

                for top_k in top_k_list:
                    print(f" Retrieval with top_k = {top_k}")

                    reference_indices = [None] * len(query_embeddings)

                    for offset in range(step_size):
                        group_indices = list(range(offset, len(query_embeddings), step_size))
                        if len(group_indices) <= top_k:
                            continue

                        for i, global_i in enumerate(group_indices):
                            # if reference_symbols[global_i] == query_symbol:
                                query_vec = query_embeddings[global_i].tolist()

                                hits = client.search(
                                    collection_name="reference_vectors",
                                    query_vector=query_vec,
                                    limit=top_k + 1,
                                    with_payload=True
                                )

                                filtered_hits = [hit for hit in hits if hit.id != global_i][:top_k]

                                for hit in filtered_hits:
                                    symbol = hit.payload["symbol"]
                                    date_key = hit.payload["date_key"]
                                    date_str = date_key.split("_")[1]
                                    date_obj = datetime.strptime(date_str, "%Y%m%d")

                                    df_symbol = df[df['Symbol'] == symbol].copy()
                                    # use MinMaxScaler
                                    
                                    total_len = len(df_symbol)
                                    split_idx = int(total_len * 0.7)
                                    train_data = df_symbol.iloc[:split_idx]
                                    test_data = df_symbol.iloc[split_idx:]

                                    scaler = MinMaxScaler()
                                    scaler.fit(train_data[['Close']])

                                    # # Transform both train and test
                                    # df_symbol.loc[:split_idx-1, 'Close'] = scaler.transform(train_data[['Close']]).flatten()
                                    # df_symbol.loc[split_idx:, 'Close'] = scaler.transform(test_data[['Close']]).flatten()
                                    # # end Minmaxscaler
                                    df_before = df_symbol[df_symbol['Date'] < date_obj].sort_values(by='Date', ascending=False).head(seq_len - 1)
                                    df_before = df_before.sort_values(by='Date')  
                                    df_current = df_symbol[df_symbol['Date'] == date_obj]
                                    df_after = df_symbol[df_symbol['Date'] > date_obj].sort_values(by='Date').head(seq_len)
                                    df_window = pd.concat([df_before, df_current, df_after]).sort_values(by='Date')

                                    close_values = torch.tensor(df_window['Close'].values, dtype=torch.float32)
                                    if reference_indices[global_i] is None:
                                        reference_indices[global_i] = close_values
                                    else:
                                        reference_indices[global_i] = torch.cat((reference_indices[global_i], close_values), dim=0)

                    reference_tensor = torch.cat(reference_indices, dim=0)

                    output_dir = f"ENCODE_IMAGE_UNET/{run_type}/{model_name}/{query_symbol}/{convert_method}/{seq_len + pred_len}"
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"{output_dir}/{top_k}_{step_size}.pt"
                    torch.save(reference_tensor, filename)
                    print(f"✅ Saved {filename} with shape {reference_tensor.shape}")

if __name__ == "__main__":
    main()
