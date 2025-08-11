import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy, csv, wandb
# wandb.login(key = '4c057ed43aa147417d2e021d9edcd9aa80cdb82e')
import torch
import numpy as np
import torch.multiprocessing
import logging
import torch.nn.functional as F
from utils.tools import EarlyStopping

from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags, get_x_and_mask
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_cond
import time
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # --- set-up data and device ---
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

        train_loader, test_loader = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        # wandb.init(entity="binh_Unet",project=f"Unet_50_{args.epochs}patience{args.epochs}_{args.symbols}", name="Binh")
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        import os
        path = os.path.join('checkpoints', args.symbols)
        if not os.path.exists(path):
            os.makedirs(path)

        model = ImagenTime(args=args, device=args.device).to(args.device)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        state = dict(model=model, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None # load ema model if available
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        best_score = {args.symbols: float('inf')}
        best_mae = {args.symbols: float('inf')}

        for epoch in range(init_epoch, args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            model.train()
            model.epoch = epoch
            train_losses = []
            eval_losses = []

            # --- train loop ---
            for i, data in enumerate(train_loader, 0):
                mask_ts, x_ts = get_x_and_mask(args, data)

                # transform to image
                x_ts_img = model.ts_to_img(x_ts)
                # pad mask with 1
                mask_ts_img = model.ts_to_img(mask_ts,pad_val=1)
                optimizer.zero_grad()
                loss = model.loss_fn_impute(x_ts_img, mask_ts_img)
                if len(loss) == 2:
                    loss, to_log = loss
                    # for key, value in to_log.items():
                    #     logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()
                train_losses.append(loss.item())
            avg_train_loss = sum(train_losses) / len(train_losses)
            # wandb.log({"train/loss": avg_train_loss}, step=epoch)
            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                mse = 0
                mae = 0
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        for idx, data in enumerate(test_loader, 1):
                            mask_ts, x_ts = get_x_and_mask(args, data)

                            # Chuyển mask và x_ts sang device thống nhất
                            device = x_ts.device
                            mask_ts = mask_ts.to(device)
                            x_ts = x_ts.to(device)

                            # transform to image
                            x_ts_img = model.ts_to_img(x_ts)
                            mask_ts_img = model.ts_to_img(mask_ts, pad_val=1)

                            # sample from the model
                            x_img_sampled = process.interpolate(x_ts_img, mask_ts_img).to(device)
                            x_ts_sampled = model.img_to_ts(x_img_sampled).to(device)

                            # task evaluation
                            x_ts = x_ts.squeeze(0)
                            x_ts_sampled = x_ts_sampled.squeeze(0)
                            mask = (mask_ts.squeeze(0) == 0)

                            mse_mean = F.mse_loss(x_ts[mask], x_ts_sampled[mask])
                            mae_mean = F.l1_loss(x_ts[mask], x_ts_sampled[mask])
                            mse += mse_mean.item()
                            mae += mae_mean.item()

                scores = {'mse': mse / (idx + 1), 'mae': mae / (idx + 1)}
                eval_losses.append(scores['mse']) # use for wandb
                vali_loss = scores['mse'] # use for Early stopping
                print(f"Epoch {epoch}, MSE: {scores['mse']}, MAE: {scores['mae']}")
                # wandb.log({
                #     "val/mse": scores['mse'],
                #     "val/mae": scores['mae']
                # }, step=epoch)
                # for key, value in scores.items():
                #     logger.log(f'test/{key}', value, epoch)
                
                early_stopping(vali_loss, model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                # --- save checkpoint ---
                curr_score = scores['mse']
                if curr_score < best_score[args.symbols]:
                    best_score[args.symbols] = curr_score
                    best_mae[args.symbols] = scores['mae']
                    ema_model = model.model_ema if args.ema else None
                    # save_checkpoint(args.log_dir, state, epoch, ema_model)

        # wandb.finish()
        print(f"Best test MSE for {args.symbols}: {best_score[args.symbols]:.4f}")
        print(f"Corresponding MAE for {args.symbols}: {best_mae[args.symbols]:.4f}")

        print(f"{args.symbols} ,{args.seq_len} , Best MSE: {best_score}, Best MAE: {best_mae}")
        
        #csv_path = os.path.join('/home/user11/thongt/ImagenTime_New_flow_3/logs', 'best_scores.csv')

        import pandas as pd
        filename_csv = "result_7_8.csv"
        from datetime import datetime 
        data = {
        "ablation": ["down-block"],
        "seed": [args.seed],
        "diffusion steps": [args.diffusion_steps],
        "symbols": [args.symbols],
        
        'ts2img( Unet flow)': ['Delay Embedding'],
        'delay': [args.delay],
        'embedding': [args.embedding],
        'diffusion_steps': [args.diffusion_steps],
        'history len': [args.seq_len // 2],
        'pred len': [args.seq_len // 2],
        'batch_size': [args.batch_size],
        
        'epochs': [args.epochs], 
        'Best_MSE': [best_score[args.symbols]],
        'Best_MAE': [best_mae[args.symbols]],
        'unet_channels': [args.unet_channels], 
        'attn_resolution': [args.attn_resolution], 
        'ch_mult': [args.ch_mult],
        'img_resolution': [args.img_resolution],
        'input_channels': [args.input_channels],
        'patience': [args.patience],
        'learning_rate': [args.learning_rate],
        'weight_decay': [args.weight_decay],  
        'time_run': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }


        df = pd.DataFrame(data)
        if os.path.exists(filename_csv):
            df.to_csv(filename_csv, mode='a', index=False, header=False)
        else:
            df.to_csv(filename_csv, index=False)

if __name__ == '__main__':
    args = parse_args_cond()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)