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
                        # wandb.init(project=f"All_Optimize_Skip_Decoder_Unet_report_{args.epochs}_{args.top_k}_{step_size}", name="Binh")
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
                        
                        reference = f"/home/user11/binhnkt/ImagenTime_New_flow_3_Backup/ENCODE_IMAGE_UNET/{local_args.run_type}/{local_args.model_name}/{local_args.symbols}/{local_args.convert_method}/{local_args.seq_len}/{local_args.top_k}_{local_args.step_size}.pt"
                        # reference = f"/home/user11/binhnkt/ImagenTime_New_flow_3_Backup/NORM_ENCODE_IMAGE_UNET/only/ViT-B-32/GOOG/gasf_gadf_linear_trend/40/{local_args.top_k}_{local_args.step_size}.pt"
                        ref = torch.load(reference)
                        # print(f"Shape of ref: {ref.shape}")

                        
                        model = ImagenTime(args=local_args, device=local_args.device).to(local_args.device)

                        # model = ImagenTime(args=args, device=args.device).to(args.device)

                        # optimizer
                        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

                        state = dict(model=model, epoch=0)
                        init_epoch = 0

                        # restore checkpoint
                        if args.resume:
                            ema_model = model.model_ema if args.ema else None # load ema model if available
                            init_epoch = restore_state(args, state, ema_model=ema_model)

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
                                # if i == 1:
                                #     break
                                batch = len(data[0])  # because last batch can not enough samples (!= batch size)
                                
                                mask_ts, x_ts = get_x_and_mask(args, data) 
                                x_ref = torch.zeros((batch, args.seq_len, args.top_k), device=args.device)
                                # print(f"x_ref {x_ref}") #(32,40,3)
                                for u in range(batch):
                                    for v in range(args.top_k):

                                        x_ref[u, :, v] = ref[total_samples * args.seq_len * args.top_k + u * args.seq_len * args.top_k + v * args.seq_len :  total_samples * args.seq_len * args.top_k + u * args.seq_len * args.top_k + (v + 1) * args.seq_len].to(args.device)

                                        x_ref[u, :, v] = x_ref[u, :, v] - args.mean
                                        x_ref[u, :, v] = x_ref[u, :, v] / args.std   

                                batch, seq_len, top_k = x_ref.shape    # #([32, 40, 3])
                                half = seq_len // 2
# 2. x_hist/fut kích thước /2
                                # x_hist = torch.zeros_like(x_ref)
                                # x_hist[:, :half, :] = x_ref[:, :half, :]   #([32, 40, 3])
                                x_hist = x_ref[:, :half, :]
                                x_future = x_ref[:, half:, :] ##([32, 20, 3]) future

                                total_samples += batch
                                sample_img = model.ts_to_img(x_ref[:, :, 0])  # shape: (batch_size, C, H, W)
                                #print(f"Shape of sample_img: {sample_img.shape}")
                                B, C, H, W = sample_img.shape # B = batch size, C = features, H = height, W = width

                                # x_ref_ts_img = torch.zeros((batch, args.top_k, H, W), device=args.device)
                                x_ref_hist_img = torch.zeros((batch, args.top_k, H, W), device=args.device)
                                x_ref_future_img = torch.zeros((batch, args.top_k, H, W), device=args.device)

                                # Gán từng ảnh transform vào
                                for j in range(args.top_k):
                                    # x_ref_ts_img[:, j] = model.ts_to_img(x_ref[:, :, j]).squeeze(1)  
                                    x_ref_hist_img[:, j] = model.ts_to_img(x_hist[:, :, j]).squeeze(1)  #([32, 3, 16, 16]) 
                                    x_ref_future_img[:, j] = model.ts_to_img(x_future[:, :, j]).squeeze(1) 


                                # transform to image
                                x_ts_img = model.ts_to_img(x_ts)     
                                torch.set_printoptions(threshold=float('inf'), precision=3, linewidth=200)
                                # print(f"x_ts {x_ts_img.shape} ")
                                # pad mask with 1
                                mask_ts_img = model.ts_to_img(mask_ts,pad_val=1)
                                optimizer.zero_grad()
                                
                                # create hist | mask, mask giống với x_ts
                                mask_for_hist = mask_ts_img.expand(-1, args.top_k, -1, -1) 
                                x_ref_hist_img = x_ref_hist_img * mask_for_hist
                                # loss = model.loss_fn_impute(x_ts_img, mask_ts_img, ref = x_ref_ts_img, top_k=args.top_k,epoch=epoch, num_epochs=args.epochs)
                                loss = model.loss_fn_impute(x_ts_img, mask_ts_img, ref_hist = x_ref_hist_img, ref_future = x_ref_future_img, top_k=args.top_k,epoch=epoch, num_epochs=args.epochs)
                                
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
                            with torch.no_grad():
                                with model.ema_scope():
                                    process = DiffusionProcess(args, model.net,
                                                            (args.input_channels, args.img_resolution, args.img_resolution))
                                    j = len(train_loader)
                                    for idx, data in enumerate(test_loader, 0):
                                        # print shape of data
                                        #print("Shape of data: ", data.shape)
                                        # if j == 1:
                                        #     break
                                        batch = len(data[0])
                

                                        mask_ts, x_ts = get_x_and_mask(args, data)

                                        x_ref = torch.zeros((batch, args.seq_len, args.top_k), device=args.device)
                                        for u in range(batch):
                                            for v in range(args.top_k):
                                                # print("chỉ số: ",idx + j) #99
                                                # print("batch size: ", args.batch_size) #8 
                                                # print((idx + j) * args.batch_size * args.seq_len * args.top_k + u * args.seq_len * args.top_k + v * args.seq_len)
                                                
                                                #print(i * args.batch_size * args.seq_len * args.seq_len * args.top_k + u * args.seq_len * args.seq_len * args.top_k + (v + 1) * args.seq_len)
                                                # add normalize
                                                x_ref[u, :, v] = ref[total_samples * args.seq_len * args.top_k + u * args.seq_len * args.top_k + v * args.seq_len :  total_samples * args.seq_len * args.top_k + u * args.seq_len * args.top_k + (v + 1) * args.seq_len].to(args.device)
                                                x_ref[u, :, v] = x_ref[u, :, v] - args.mean                                        
                                                x_ref[u, :, v] = x_ref[u, :, v] / args.std                                   
                                        batch, seq_len, top_k = x_ref.shape  # 32,40,3
                                        half = seq_len // 2

                                        # # Tạo tensor toàn 0 cùng shape
                                        # x_hist = torch.zeros_like(x_ref)
                                        # x_future = torch.zeros_like(x_ref)
                                        # # Gán nửa đầu cho x_hist
                                        # x_hist[:, :half, :] = x_ref[:, :half, :]
                                        # # Gán nửa sau cho x_future
                                        # x_future[:, :seq_len - half, :] = x_ref[:, half:, :]
                                        # x_hist = x_ref[:, :half, :]
                                        # x_future = x_ref[:, half:, :] 

#20 20 
                                        # x_hist = torch.zeros_like(x_ref)
                                        # x_hist[:, :half, :] = x_ref[:, :half, :]   #([32, 20, 3])
                                        x_hist = x_ref[:, :half, :]
                                        x_future = x_ref[:, half:, :] 
#40 40 
                                        # x_hist = x_ref    #([32, 20, 3])
                                        # x_future = x_ref

                                        total_samples += batch
                                        sample_img = model.ts_to_img(x_ref[:, :, 0])  # shape: (batch_size, C, H, W)
                                        #print(f"Shape of sample_img: {sample_img.shape}")
                                        B, C, H, W = sample_img.shape # B = batch size, C = features, H = height, W = width
                                        # x_ref_ts_img = torch.zeros((batch, args.top_k, H, W), device=args.device)
                                        x_ref_hist_img = torch.zeros((batch, args.top_k, H, W), device=args.device)
                                        x_ref_future_img = torch.zeros((batch, args.top_k, H, W), device=args.device)

                                        # Gán từng ảnh transform vào
                                        for i in range(args.top_k):
                                            # x_ref_ts_img[:, i] = model.ts_to_img(x_ref[:, :, i]).squeeze(1)
                                            # print(f"Shape của x_hist bên trong test loader: {x_hist.shape}")
                                            x_ref_hist_img[:, i] = model.ts_to_img(x_hist[:, :, i]).squeeze(1) 
                                            x_ref_future_img[:, i] = model.ts_to_img(x_future[:, :, i]).squeeze(1) 
                                        #print(f"Shape of x_ref_ts_img: {x_ref_ts_img.shape}")
                                        x_ts_img = model.ts_to_img(x_ts)
                                        mask_ts_img = model.ts_to_img(mask_ts, pad_val=1)

                                        mask_for_hist = mask_ts_img.expand(-1, args.top_k, -1, -1) 
                                        x_ref_hist_img = x_ref_hist_img * mask_for_hist
                                        # sample from the model
                                        # and impute, both interpolation and extrapolation are similar just the mask is different
                                        # x_img_sampled = process.interpolate(x_ts_img, mask_ts_img, ref=x_ref_ts_img).to(x_ts_img.device)

                                        x_img_sampled = process.interpolate(x_ts_img, mask_ts_img, ref_hist = x_ref_hist_img,ref_future = x_ref_future_img).to(x_ts_img.device)
                                        x_ts_sampled = model.img_to_ts(x_img_sampled)

                                        # task evaluation
                                        x_ts= x_ts.squeeze(0)
                                        x_ts_sampled = x_ts_sampled.squeeze(0)
                                        #print(f"Shape of x_ts: {x_ts.shape}, x_ts_sampled: {x_ts_sampled.shape}, mask_ts: {mask_ts.shape}")
                                        mse_mean = F.mse_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0])
                                        
                                        mae_mean = F.l1_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0])
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
                                print(f"🟢 New best at epoch {epoch}, top k {local_args.top_k}, step size {step_size}: MSE={best_score_mse:.4f}, MAE={best_score_mae:.4f}")
                                ema_model = model.model_ema if args.ema else None
                                save_checkpoint(args.log_dir, state, epoch, ema_model)
                        
                        # wandb.finish()
                        
                        print(f"Top k: {local_args.top_k}, Step size: {local_args.step_size}, Best MSE: {best_score_mse}, Best MAE: {best_score_mae}")
                        
                        #csv_path = os.path.join('/home/user11/thongt/ImagenTime_New_flow_3/logs', 'best_scores.csv')
                        print('hello')
                        import pandas as pd
                        filename_csv = "logs/New_Skip_Decoder_Unet_report_new.csv"

                        data = {
                            "seed": [local_args.seed],
                            "diffusion steps": [local_args.diffusion_steps],
                            "symbols": [local_args.symbols],
                            "database": [local_args.run_type],
                            'CLIP model': [local_args.model_name],
                            'ts2img (retrieval)':[local_args.convert_method],
                            'ts2img( Unet flow)': 'Delay Embedding',
                            'history len': [local_args.seq_len // 2],
                            'pred len':[local_args.seq_len // 2],
                            'top k': [local_args.top_k], 
                            'step size': [local_args.step_size],
                            'batch size': [args.batch_size], 
                            'epochs': [local_args.epochs], 
                            'Best_MSE': [best_score_mse],
                            'Best_MAE': [best_score_mae],
                            'unet_channels': [args.unet_channels], 
                            'attn_resolution': [args.attn_resolution], 
                            'ch_mult': [args.ch_mult],
                            'img_resolution': [args.img_resolution],
                            'input_channels': [args.input_channels],
                            'patience': [args.patience],
                            'learning_rate': [args.learning_rate],
                            'weight_decay': [args.weight_decay],                           
                        }

                        df = pd.DataFrame(data)
                        if os.path.exists(filename_csv):
                            df.to_csv(filename_csv, mode='a', index=False, header=False)
                        else:
                            df.to_csv(filename_csv, index=False)



                        # with open(csv_path, mode='w', newline='') as file:
                        #     writer = csv.writer(file)
                                            

                        #     writer.writerow(['seed', 'Symbol', 'Database', 'CLIP model', 'ts2img (retrieval)', 'ts2img( Unet flow)', 'history len', 'pred len', 'top k', 'step size', 'batch size', 'epochs', 'Best_MSE', 'Best_MAE'])
                        #     writer.writerow([local_args.seed, local_args.symbols, local_args.run_type, local_args.model_name, local_args.convert_method, 'Delay embedding', local_args.seq_len //2, local_args.seq_len // 2, local_args.top_k, local_args.step_size, 32, local_args.epochs, best_score_mse, best_score_mae])
                
    logging.info("Training is complete")
if __name__ == '__main__':
    args = parse_args_cond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)