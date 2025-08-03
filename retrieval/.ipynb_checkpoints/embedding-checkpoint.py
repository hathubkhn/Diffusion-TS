import pandas as pd
import numpy as np
import os
import pickle
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LinearRegression
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


seq_len_list = [60] #[20, 60]
symbol_list = ['GOOG'] #, 'AAPL',  'AMZN'
database_list = ['only'] #['only', 'industry', 'all']
from torch.nn.functional import cosine_similarity

import torchvision.models as models
import torch.nn as nn
import random
import torch
from torchvision.models import vgg16, VGG16_Weights

def init_model():
		weights = VGG16_Weights.DEFAULT  # hoặc .IMAGENET1K_V1
		vgg16_model = vgg16(weights=weights).eval()
		first_layers = nn.Sequential(*list(vgg16_model.features.children())[:3])
		
		return first_layers

def embedding(input_tensor, first_layers=None):
    
    output_tensor = first_layers(input_tensor)
    # Max pooling
    #output_tensor = nn.MaxPool2d(kernel_size=2, stride=2)(output_tensor)
    # Flatten
    output_tensor = output_tensor.view(output_tensor.size(0), -1)
    return output_tensor

first_layers = init_model()


for seq_len in seq_len_list:
  for symbol in symbol_list:
    for database in database_list:
      total_len = 2 * seq_len  # Total length of the sequence
      if database == 'only' or database == 'industry':
          input_file = f'/home/user11/thongt/Diffusion-TS-all-stock/data_new/{symbol}_{database}_1.csv'
      if database == 'all':
          input_file = f'/home/user11/thongt/Diffusion-TS-all-stock/data_new/all_1.csv'
    
        
      output_dir = f'/home/user11/thongt/ImagenTime_New_flow_3_Backup_2_shuffle/TS2IMG_UNET_1/{symbol}/{symbol}_{database}_gasf_gadf_linear_trend/{seq_len}'
      os.makedirs(output_dir, exist_ok=True)

      df = pd.read_csv(input_file, usecols=['Date', 'Symbol', 'Close'])
      df['Date'] = pd.to_datetime(df['Date'])
      df = df.sort_values(['Symbol', 'Date']) 

      df['Close'] = df['Close'].astype(float)
      all_sequences = []

      def compute_linear_trend(window):
          x = np.arange(len(window)).reshape(-1, 1)
          y = window.reshape(-1, 1)
          
          model = LinearRegression()
          model.fit(x, y)
          trend = model.predict(x).flatten()
          return trend

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
                
              stacked_scaled = ((stacked - np.min(stacked)) / (np.max(stacked) - np.min(stacked)) * 255).astype(np.uint8)
              tensor_img = torch.tensor(stacked_scaled.transpose(2, 0, 1), dtype=torch.float32)
              
              #embedding_img = embedding(tensor_img, first_layers=first_layers)
              tensor_images.append(tensor_img)
          batch_tensor = torch.stack(tensor_images, dim=0)


            	
              
						

          # phần tử đầu
      ref_tensor = batch_tensor[-1].unsqueeze(0).repeat(batch_tensor.shape[0] - 60, 1, 1, 1)
      ref_emb = embedding(ref_tensor, first_layers=first_layers)
      cmp_tensor = batch_tensor[:-60]
      cmp_emb = embedding(cmp_tensor, first_layers=first_layers)
      print(ref_emb.shape)
      print(cmp_tensor.shape)

		  
      #  SSIM scores
      # metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None)
      # metric.update(ref_tensor, cmp_tensor)
      # ssim_scores = metric.compute()
      similarities = cosine_similarity(ref_emb, cmp_emb, dim=1)
      
      top_k = 10
      top_scores, top_indices = torch.topk(similarities, k=top_k)
      print(top_scores)
      #top_windows = [window_list[i] for i in top_indices]
      
          
              
      

      print("OK")
