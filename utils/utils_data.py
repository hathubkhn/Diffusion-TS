import numpy as np
import torchaudio.transforms as transforms
import os
import pandas as pd
import sys
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler as Ori_MinMaxScaler
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset
from data.data_provider.data_factory import data_provider
# from data.long_range import parse_datasets 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def real_data_loading(args, data_name, seq_len):
    """Load and preprocess real-world data.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    #assert data_name in ['goog', 'amzn', 'aapl', 'energy', 'metro','jpm', 'nee']

    # if data_name == 'goog':
    #     ori_data = np.loadtxt('./data/short_range/GOOG.csv', delimiter=",", skiprows=1)
    #     # ori_data = pd.read_csv('./data/short_range/GOOG.csv', delimiter=",").values
    #     #ori_data = np.genfromtxt('./data/short_range/GOOG.csv', delimiter=",", skip_header=1, dtype=None, encoding='utf-8')
    # elif data_name == 'aapl':
    #     ori_data = np.loadtxt('./data/short_range/AAPL.csv', delimiter=",", skiprows=1)
    # elif data_name == 'amzn':
    #     ori_data = np.loadtxt('./data/short_range/AMZN.csv', delimiter=",", skiprows=1)
    # elif data_name == 'energy':
    #     ori_data = np.loadtxt('./data/short_range/energy_data.csv', delimiter=",", skiprows=1)
    # elif data_name == 'metro':
    #     ori_data = np.loadtxt('./data/short_range/metro_data.csv', delimiter=",", skiprows=1)

    data_name_upper = data_name.upper()
    ori_data = np.loadtxt(f'./data/short_range/{data_name_upper}.csv', delimiter=",", skiprows=1)


    # Flip the data to make chronological data
    #ori_data = ori_data[::-1]

    ori_data = torch.Tensor(ori_data)  # shape [N]

    train_ratio = 0.7
    train_size = int(len(ori_data) * train_ratio)
    train_data = ori_data[:train_size]
    test_data = ori_data[train_size:]

    scaler = Ori_MinMaxScaler()  #  StandardScaler()
    
    train_data = scaler.fit_transform(train_data.reshape(-1, 1)).reshape(train_data.shape)
    test_data = scaler.transform(test_data.reshape(-1, 1)).reshape(test_data.shape)
    

    
    #ori_data = torch.cat((train_data, test_data), dim=0)
    # Save mean and std 
    #mean, std = scaler.mean_, scaler.var_
    mean, std = scaler.data_min_, scaler.data_max_ - scaler.data_min_
    args.mean, args.std = torch.Tensor(mean), torch.Tensor(std)   
    args.mean, args.std = args.mean.to(args.device), args.std.to(args.device)

    train_set = []
    test_set = []
    # Cut data by sequence length
    for i in range(0, len(train_data) - seq_len + 1):
        _x = train_data[i:i + seq_len]
        train_set.append(_x)
    for i in range(0, len(test_data) - seq_len + 1):
        _x = test_data[i:i + seq_len]
        test_set.append(_x)

    return train_set, test_set

# def normalize(data):
#     numerator = data - np.min(data, 0)
#     denominator = np.max(data, 0) - np.min(data, 0)
#     norm_data = numerator / (denominator + 1e-7)
#     return norm_data

def normalize(data, mean=None, std=None):
    return (data - mean) / (std + 1e-7)
def denormalize(data, mean, std):
    return data * std + mean

def gen_dataloader(args):
    if args.dataset == 'sine':
        args.dataset_size = 10000
        ori_data = sine_data_generation(args.dataset_size, args.seq_len, args.input_channels)
        ori_data = torch.Tensor(np.array(ori_data))
        train_set = Data.TensorDataset(ori_data)

    elif args.dataset in ['goog', 'amzn', 'aapl', 'energy','jpm', 'nee','xom', 'pg']:
        train_data, test_data = real_data_loading(args, args.dataset, args.seq_len)
        # reference = f"/home/user11/thongt/ImagenTime_New_flow_3_Backup_2/ENCODE_IMAGE_UNET/{args.run_type}/ViT-B-32/{args.symbols}/gasf_gadf_linear_trend/{args.seq_len}/{args.top_k}_10.pt" #{args.model_name}, {args.convert_method}, {args.step_sizes}
        reference = f"/home/user11/thongt/ImagenTime_New_flow_3_Backup_2_shuffle/Database/{args.symbols}/{args.pretrained_model}/{args.num_first_layer}/{args.run_type}_gasf_gadf/{args.seq_len // 2}.pt"
        ref = torch.load(reference)
        print(ref.shape)
        
        train_data = torch.Tensor(np.array(train_data))
        test_data = torch.Tensor(np.array(test_data))
        num_train = len(train_data)
        num_test = len(test_data)
        train_ref_data = []
        test_ref_data = []
        for i in range(num_train):
            ref_i_top10 = ref[i * 10 * args.seq_len: (i + 1) * 10 * args.seq_len]
            ref_i = ref_i_top10[0 : args.top_k * args.seq_len]
            ref_i = torch.tensor(ref_i, dtype=torch.float32)
            train_ref_data.append(torch.cat([train_data[i], ref_i], dim=0))
                            

        # có seq_len - 1 sample giữa train và test không được dùng nên ko load reference
        for i in range(num_test):
            index = i + num_train + args.seq_len - 1
            ref_i_top10 = ref[index * 10 * args.seq_len: (index + 1) * 10 * args.seq_len]
            if i == num_test - 1:
                print((index + 1) * 10 * args.seq_len)
            ref_i = ref_i_top10[0 : args.top_k * args.seq_len]

            ref_i = torch.tensor(ref_i, dtype=torch.float32)
            test_ref_data.append(torch.cat([test_data[i], ref_i], dim=0))



        # create TensorDataset and DataLoader
        train_ref_data = torch.stack(train_ref_data)
        test_ref_data = torch.stack(test_ref_data)
        train_set = Data.TensorDataset(train_ref_data)
        test_set = Data.TensorDataset(test_ref_data)
        
        train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=False)

        test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False,
                                num_workers= args.num_workers, drop_last=False)

        

        return train_loader, test_loader

    elif args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        train_data, train_loader = data_provider(args, flag='train')
        test_data, test_loader = data_provider(args, flag='test')
        return train_loader, test_loader

    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)

    # for the short-term time series benchmark, the entire dataset for both training and testing
    return train_loader, train_loader





def stft_transform(data, args):
    data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
    n_fft = args.n_fft
    hop_length = args.hop_length
    spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True, power=None)
    transformed_data = spec(data)
    real, min_real, max_real = MinMaxScaler(transformed_data.real.numpy(), True)
    real = (real - 0.5) * 2
    imag, min_imag, max_imag = MinMaxScaler(transformed_data.imag.numpy(), True)
    imag = (imag - 0.5) * 2
    # saving min and max values, we will need them for inverse transform
    args.min_real, args.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
    args.min_imag, args.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)
    return torch.Tensor(real), torch.tensor(imag)


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


