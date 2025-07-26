import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy, csv, wandb
# wandb.login(key = 'c9bf0410a696f6094571b4bcf35e7f93c95fe86d')
import torch
import numpy as np
import torch.multiprocessing
import logging, copy
import torch.nn.functional as F
from utils.tools import EarlyStopping
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags, get_x_and_mask
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_cond
# from utils.utils_vis import visual

#torch.multiprocessing.set_sharing_strategy('file_system')


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
        symbols = args.symbols
        model_name = args.model_name
        run_type = args.run_type
        top_k = args.top_k
        step_size = args.step_sizes
        convert_method = args.convert_method
        train_loader, test_loader = gen_dataloader(args)  
        print(f"Train loader length: {len(train_loader)}, Test loader length: {len(test_loader)}")

        logging.info(args.dataset + ' dataset is ready.')
        
        for model_name in args.model_name:
            for convert_method in args.convert_method:
                
                    for step_size in args.step_sizes:
                        # wandb.init(project=f"All_Optimize_Skip_Decoder_Unet_report_{args.epochs}_{args.top_k}_{step_size}", name="Thong")
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
                        path = os.path.join('checkpoints', args.symbols)
                        if not os.path.exists(path):
                            os.makedirs(path)

                        # update args
                        local_args = copy.deepcopy(args)
                          # (batch_size, seq_len, top_k)


                        # update local_args with specific parameters
        
                        local_args.model_name = model_name
                        local_args.convert_method = convert_method
                        local_args.top_k = top_k
                        local_args.step_size = step_size
                        
                        reference = f"./ENCODE_IMAGE_UNET/{local_args.run_type}/{local_args.model_name}/{local_args.symbols}/{local_args.convert_method}/{local_args.seq_len}/{local_args.top_k}_{local_args.step_size}.pt"
                        ref = torch.load(reference)
                        print(f"Shape of ref: {ref.shape}")

                        
                        model = ImagenTime(args=local_args, device=local_args.device).to(local_args.device)

                        # optimizer
                        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

                        state = dict(model=model, epoch=0)
                        init_epoch = 0

                        # restore checkpoint
                        if args.resume:
                            ema_model = model.model_ema if args.ema else None # load ema model if available
                            init_epoch = restore_state(args, state, ema_model=ema_model)

                        # print model parameters
                        #print_model_params(logger, model)

                        # --- train model ---
                        logging.info(f"Continuing training loop from epoch {init_epoch}.")
                        
                        best_score_mae = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
                        best_score_mse = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
                        for epoch in range(init_epoch, args.epochs):
                            print(f"Epoch {epoch + 1}/{args.epochs}")
                            model.train()
                            model.epoch = epoch
                            train_losses = []
                            eval_losses = []

                            # logger.log_name_params('train/epoch', epoch)

                            # --- train loop ---
                            total_samples = 0 # the number of sample used
                            for i, data in enumerate(train_loader, 0): #1
                                
                                batch = len(data[0])  # because last batch can not enough samples (!= batch size)
                                
                                x_ts = data[0].float().to(args.device)
                                # print("11111")
                                # half ones and half zeros
                                mask_ts = torch.zeros_like(x_ts)
                                mask_ts[:, :x_ts.shape[1] // 2] = 1
                                x_ref = torch.zeros((batch, args.seq_len, args.top_k), device=args.device)
                                # print(f"x_ref {x_ref}") #(32,40,3)
                                for u in range(batch):
                                    for v in range(args.top_k):

                                        x_ref[u, :, v] = ref[total_samples * args.seq_len * args.top_k + u * args.seq_len * args.top_k + v * args.seq_len :  total_samples * args.seq_len * args.top_k + u * args.seq_len * args.top_k + (v + 1) * args.seq_len].to(args.device)
                                        #Normalize the x_ref
                                        x_ref[u, :, v] = x_ref[u, :, v] - args.mean
                                        x_ref[u, :, v] = x_ref[u, :, v] / args.std

                                total_samples += batch
                                sample_img = model.ts_to_img(x_ref[:, :, 0])  # shape: (batch_size, C, H, W)
            
                                B, C, H, W = sample_img.shape # B = batch size, C = features, H = height, W = width
                                x_ref_ts_img = torch.zeros((batch, args.top_k, H, W), device=args.device)

                                # ref transform ts to img
                                for j in range(args.top_k):
                                    x_ref_ts_img[:, j] = model.ts_to_img(x_ref[:, :, j]).squeeze(1)
                                # transform to image
                                x_ts_img = model.ts_to_img(x_ts)
                                # pad mask with 1
                                mask_ts_img = model.ts_to_img(mask_ts,pad_val=1)
                                optimizer.zero_grad()  
                                loss = model.loss_fn_impute(x_ts_img, mask_ts_img, ref = x_ref_ts_img, top_k=args.top_k)
                                
                                if len(loss) == 2:
                                    loss, to_log = loss
                                    for key, value in to_log.items():
                                        logger.log(f'train/{key}', value, epoch)
                                

                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                                optimizer.step()
                                model.on_train_batch_end()
                                train_losses.append(loss.item())
                            
                            avg_train_loss = sum(train_losses) / len(train_losses)
                            # wandb.log({"train/loss": avg_train_loss}, step=epoch)
                                

                            # --- evaluation loop ---
                            # best_score_mae = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
                            # best_score_mse = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
                            #if epoch % args.logging_iter == 0:
                            mse = 0
                            mae = 0
                            model.eval()
                            total_samples += (local_args.seq_len - 1)  # skip some elements that are interaction between train and test sets
                            with torch.no_grad():
                                with model.ema_scope():
                                    process = DiffusionProcess(args, model.net,
                                                            (args.input_channels, args.img_resolution, args.img_resolution))
                                    
                                    j = len(train_loader)
                                    
                                    
                                    print("len(test_loader):", len(test_loader))
                                    
                                    for idx, data in enumerate(test_loader, 0): 
                                        print(f"Test batch {idx + 1}/{len(test_loader)}")
                                        # if idx == 0:
                                        #     break
                                        # data_1 = copy.deepcopy(data)
                                        # mean = mean.to(data_1[0].device)
                                        # std = std.to(data_1[0].device)
                                        # data_1[0] = ((data_1[0] - mean) / std)  # normalize data
                                       
                                        batch = len(data[0])
    
                                        folder_path = f'./test_results/{local_args.symbols}{local_args.top_k}{local_args.step_size}/'
                                        if not os.path.exists(folder_path):
                                            os.makedirs(folder_path)
                
                                        # _, x_ts_copy = get_x_and_mask(args, data)
                                        # mask_ts, x_ts = get_x_and_mask(args, data_1)

                                        mask_ts, x_ts = get_x_and_mask(args, data)
                                        
                                        #print(f"Shape of mask_ts: {mask_ts.shape}, x_ts: {x_ts.shape}") # (32 40)
                                        
                                        x_ref = torch.zeros((batch, args.seq_len, args.top_k), device=args.device)
                                        for u in range(batch):
                                            for v in range(args.top_k):
                                                
                                                x_ref[u, :, v] = ref[total_samples * args.seq_len * args.top_k + u * args.seq_len * args.top_k + v * args.seq_len :  total_samples * args.seq_len * args.top_k + u * args.seq_len * args.top_k + (v + 1) * args.seq_len].to(args.device)
                                                #Normalize the x_ref
                                                x_ref[u, :, v] = x_ref[u, :, v] - args.mean
                                                x_ref[u, :, v] = x_ref[u, :, v] / args.std

                                                
                                        
                                        total_samples += batch
                                        sample_img = model.ts_to_img(x_ref[:, :, 0])  # shape: (batch_size, C, H, W)
                                        #print(f"Shape of sample_img: {sample_img.shape}")
                                        B, C, H, W = sample_img.shape # B = batch size, C = features, H = height, W = width
                                        x_ref_ts_img = torch.zeros((batch, args.top_k, H, W), device=args.device)

                                        for i in range(args.top_k):
                                            x_ref_ts_img[:, i] = model.ts_to_img(x_ref[:, :, i]).squeeze(1)
                                        #print(f"Shape of x_ref_ts_img: {x_ref_ts_img.shape}")
                                        x_ts_img = model.ts_to_img(x_ts)
                                        mask_ts_img = model.ts_to_img(mask_ts, pad_val=1)
                                    

                                        # sample from the model
                                        # and impute, both interpolation and extrapolation are similar just the mask is different
                                    
                                        x_img_sampled = process.interpolate(x_ts_img, mask_ts_img, ref = x_ref_ts_img).to(x_ts_img.device)
                                        x_ts_sampled = model.img_to_ts(x_img_sampled)


                                        # task evaluation
                                
                                        x_ts_pred = x_ts_sampled.squeeze(2)
                                        mse_mean = F.mse_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_pred[mask_ts == 0])
                                        
                                        mae_mean = F.l1_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_pred[mask_ts == 0])
                                        if epoch % 5 == 0:
                                            visual(x_ts[0].cpu().numpy(), x_ts_pred[0].cpu().numpy(),
                                                f"{folder_path}/epoch_{epoch}_idx_{idx}_mse_{mse_mean.item():.4f}_mae_{mae_mean.item():.4f}.png")
                                               
                                                  
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
                            for key, value in scores.items():
                                logger.log(f'test/{key}', value, epoch)
                            
                            early_stopping(vali_loss, model, path)
                            if early_stopping.early_stop:
                                print("Early stopping")
                                break
                            

                            # --- save checkpoint ---
                            curr_score_mse = scores['mse']
                            curr_score_mae = scores['mae']
                            if curr_score_mse < best_score_mse:
                                best_score_mse = curr_score_mse
                                best_score_mae = curr_score_mae
                                print(f" New best at epoch {epoch}, top k {local_args.top_k}, step size {step_size}: MSE={best_score_mse:.4f}, MAE={best_score_mae:.4f}")
                                ema_model = model.model_ema if args.ema else None
                                save_checkpoint(args.log_dir, state, epoch, ema_model)
                        
                        # wandb.finish()
                        
                        print(f"Symbol: {local_args.symbols}, Seq_len: {local_args.seq_len}, Top k: {local_args.top_k}, Step size: {local_args.step_size}, Best MSE: {best_score_mse}, Best MAE: {best_score_mae}")
                        
                        print('hello')
                        import pandas as pd
                        from datetime import datetime 
                        #filename_csv = "logs/New_Skip_Decoder_Unet_report.csv"
                        filename_csv = "logs/test_17_7_batch_32.csv"
                        if not os.path.exists(filename_csv):
                            with open(filename_csv, mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(['architecture','seed', 'diffusion steps', 'symbols', 'database', 'CLIP model', 'ts2img (retrieval)', 'ts2img( Unet flow)',  'delay', 'img resolution', 'history len', 'pred len', 'top k', 'step size', 'batch size', 'epochs', 'Best_MSE', 'Best_MAE'])

                        data = {
                            "attention architecture": 'Encoder-only',
                            "seed": [local_args.seed],
                            "diffusion steps": [local_args.diffusion_steps],
                            "symbols": [local_args.symbols],
                            'top k': [local_args.top_k], 
                            'step size': [local_args.step_size],
                            "database": [local_args.run_type],
                            'CLIP model': [local_args.model_name],
                            'ts2img (retrieval)':[local_args.convert_method],
                            'ts2img( Unet flow)': 'Delay Embedding',
                            'delay': [local_args.delay],
                            'img resolution': [local_args.img_resolution],                          
                            'history len': [local_args.seq_len // 2],
                            'pred len':[local_args.seq_len // 2],
                            'batch size': [args.batch_size], 
                            'epochs': [local_args.epochs], 
                            'unet_channels': [args.unet_channels], 
                            'attn_resolution': [args.attn_resolution], 
                            'ch_mult': [args.ch_mult],
                            'img_resolution': [args.img_resolution],
                            'input_channels': [args.input_channels],
                            'Best_MSE': [best_score_mse],
                            'Best_MAE': [best_score_mae],
                            'time_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S')       
                        }

                        df = pd.DataFrame(data)
                        if os.path.exists(filename_csv):
                            df.to_csv(filename_csv, mode='a', index=False, header=False)
                        else:
                            df.to_csv(filename_csv, index=False)

    logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_cond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)