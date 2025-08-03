from utils.utils_data import gen_dataloader
import os
import sys
import numpy as np
import torch
import torch.multiprocessing
import argparse
import time
args = argparse.Namespace()
args.dataset = 'goog'
args.seq_len = 40
args.batch_size = 32
args.num_workers = 4

torch.multiprocessing.set_sharing_strategy('file_system')

train_loader, test_loader = gen_dataloader(args)

for i, batch in enumerate(test_loader):
    print(f"[Start] Batch {i}", flush=True)
    time.sleep(10)
    print(f"[End]   Batch {i}", flush=True)

