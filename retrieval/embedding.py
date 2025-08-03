import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LinearRegression
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.nn.functional import cosine_similarity

import torchvision.models as models
import torch.nn as nn
import random
import torch
from torchvision.models import vgg16, VGG16_Weights


seq_len_list = [20, 60] #[10,20,40, 60]
symbol_list = ['GEV', 'JPM', 'LIN', 'LLY', 'NEE', 'PG', 'PLD', 'XOM'] #, 'AAPL',  'AMZN'
database_list = ['only'] #['only', 'industry', 'all']
num_first_layers = [3, 4, 5]


def init_model(num_first_layer):
		weights = VGG16_Weights.DEFAULT  # hoặc .IMAGENET1K_V1
		vgg16_model = vgg16(weights=weights).eval()
		first_layers = nn.Sequential(*list(vgg16_model.features.children())[:num_first_layer])
		return first_layers

def embedding(input_tensor, first_layers=None):
    
    output_tensor = first_layers(input_tensor)
    # Max pooling
    #output_tensor = nn.MaxPool2d(kernel_size=2, stride=2)(output_tensor)
    # Flatten
    output_tensor = output_tensor.view(-1)
    
    return output_tensor

def compute_linear_trend(window):
    x = np.arange(len(window)).reshape(-1, 1)
    y = window.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(x, y)
    trend = model.predict(x).flatten()
    return trend


for num_first_layer in num_first_layers:
    first_layers = init_model(num_first_layer=num_first_layer)
    for seq_len in seq_len_list:
        for symbol in symbol_list:
            for database in database_list:
                total_len = 2 * seq_len  # Total length of the sequence
                if database == 'only' or database == 'industry':
                    #input_file = f'/home/user11/thongt/Diffusion-TS-all-stock/data_new/{symbol}_{database}_1.csv'
                    input_file = f'/home/user11/binhnkt/data_new/{symbol}.csv'
                if database == 'all':
                    input_file = f'/home/user11/thongt/Diffusion-TS-all-stock/data_new/all_1.csv'
                

                df = pd.read_csv(input_file, usecols=['Date', 'Symbol', 'Close'])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values(['Symbol', 'Date']) 

                df['Close'] = df['Close'].astype(float)
                all_sequences = []

                
                for symbol in tqdm(df['Symbol'].unique(), desc="Xử lý symbol"):
                    print("Hello")
                    df_symbol = df[df['Symbol'] == symbol].copy()
                    close_prices = df_symbol['Close'].values
                    if len(close_prices) < total_len:
                        continue 

                    #scaler = MinMaxScaler(feature_range=(-1, 1))
                    normalized = close_prices.copy()

                    # GAF
                    gasf = GramianAngularField(image_size=seq_len, method='summation')
                    gadf = GramianAngularField(image_size=seq_len, method='difference')
                    tensor_images = []
                    window_list = []
                        
                    for i in range(len(normalized) - total_len + 1):
                        window = normalized[i:i + seq_len].reshape(1, -1)
                        # window_list.append(window.copy())
                        window_list.append(normalized[i:i + 2 * seq_len].reshape(1, -1))
                        # Generate GAF images
                        gasf_img = gasf.transform(window)[0]
                        gadf_img = gadf.transform(window)[0]
                        zeros_img = np.zeros_like(gadf_img)
                        trend = compute_linear_trend(window.flatten()).reshape(seq_len, 1)
                        trend_img = np.tile(trend, (1, seq_len))
                    
                        #stacked = np.stack([gasf_img, gadf_img, trend_img], axis=-1)
                        stacked = np.stack([gasf_img, gadf_img, zeros_img], axis=-1)
                            
                        #stacked_scaled = ((stacked - np.min(stacked)) / (np.max(stacked) - np.min(stacked)) * 255).astype(np.uint8)
                        tensor_img = torch.tensor(stacked.transpose(2, 0, 1), dtype=torch.float32)
                        
                        embedding_img = embedding(tensor_img, first_layers=first_layers)
                        tensor_images.append(embedding_img)
                    batch_emb = torch.stack(tensor_images, dim=0)
                    top_k = 10
                    margin = seq_len
                    N = len(batch_emb)
                    print(N)
                    # split train and test
                    split = int(0.7 * N)
                    
                    all_topk_windows = []
                    for i in range(N):
                        valid_indices = [j for j in range(N) if abs(j - i) > margin and j <= split]
                        if len(valid_indices) < top_k:
                            continue
                        query_emb = batch_emb[i].unsqueeze(0)
                        ref_emb = batch_emb[valid_indices]
                        similarities = cosine_similarity(query_emb, ref_emb, dim=1)                       
                        top_scores, top_indices = torch.topk(similarities, k=top_k)
                        #print(top_scores)                      
                        selected_windows = [window_list[valid_indices[j]] for j in top_indices]
                        
                        for win in selected_windows:
                            all_topk_windows.append(win.flatten())
                    concat_tensor = torch.tensor(np.stack(all_topk_windows), dtype=torch.float32).flatten()
                    print(concat_tensor.shape)
                    output_file = f'/home/user11/thongt/ImagenTime_New_flow_3_Backup_2_shuffle/Database/{symbol}/VGG16/{num_first_layer}/{database}_gasf_gadf/{seq_len}.pt'
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    torch.save(concat_tensor, output_file)
                print("OK")



            	
              