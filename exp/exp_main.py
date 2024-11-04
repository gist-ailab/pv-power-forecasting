from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, LSTM
from models.Stat_models import Naive_repeat, Arima
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, visual_out, visual_original
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import datetime

import warnings
import matplotlib.pyplot as plt
import numpy as np
import wandb
from utils.wandb_uploader import upload_files_to_wandb

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        self.project_name = "pv-forecasting"
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.args.model}_run_{current_time}"

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'Naive_repeat': Naive_repeat,
            'Arima': Arima,
            'LSTM': LSTM
        }
        if self.args.model == 'LSTM':
            model = model_dict[self.args.model].Model(self.args, self.device).float()
        else: model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, part=None):
        if part == None:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            if part == 'linear_probe':
                model_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        return model_optim


    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting, resume):

        self._set_wandb(setting)
        
        # Initialize wandb with the current settings
        config = {
            "model": self.args.model,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),  # 모델 파라미터 개수,
            "batch_size": self.args.batch_size,
            "num_workers": self.args.num_workers,
            "learning_rate": self.args.learning_rate,
            "loss_function": self.args.loss,
            "dataset": self.args.data,
            "epochs": self.args.train_epochs,
            "input_seqeunce_length": self.args.seq_len,
            "prediction_sequence_length": self.args.pred_len,
            "patch_length": self.args.patch_len,
            "stride": self.args.stride,
        }
        upload_files_to_wandb(
            project_name=self.project_name,
            run_name=self.run_name,
            config=config
        )     
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if 'checkpoint.pth' in self.args.checkpoints.split('/'):
            path = self.args.checkpoints
        else:
            path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
            
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        if resume:
            latest_model_path = path + '/' + 'model_latest.pth'
            self.model.load_state_dict(torch.load(latest_model_path))
            print('model loaded from {}'.format(latest_model_path))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, data in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark, site, batch_x_ts, batch_y_ts = data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                site = site.to(self.device)
                batch_x_ts = batch_x_ts.to(self.device)
                batch_y_ts = batch_y_ts.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                mark = True if self.args.is_pretraining else False
                if self.args.use_amp:
                   
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                            if  'LSTM' in self.args.model:
                                outputs = self.model(batch_x, dec_inp)
                            else:
                                outputs = self.model(batch_x, mark)
            
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                        if 'LSTM' in self.args.model:
                            outputs = self.model(batch_x, dec_inp)
                        else:
                            outputs = self.model(batch_x, mark)
            
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                            
                            
                f_dim = -1 if self.args.features == 'MS' else 0
                
                # loss for source domain
                outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # outputs_inv = train_data.inverse_transform_tensor(site[:, 0], outputs)
                # batch_y_inv = train_data.inverse_transform_tensor(site[:, 0], batch_y)

                # outputs_np = torch.from_numpy(outputs_np).to(self.device)
                # batch_y_np = torch.from_numpy(batch_y_np).to(self.device)

                # outputs_np.requires_grad = True
                # batch_y_np.requires_grad = True
                loss = criterion(outputs, batch_y)
                train_losses.append(loss.item())

                if (i + 1) % 100 == 0:
                    # Log iteration-level metrics every 100 iterations
                    wandb.log({
                        "iteration": (epoch * len(train_loader)) + i + 1,
                        "train_loss_iteration": loss.item()
                    })
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}, ")
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                    
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} | cost time: {time.time() - epoch_time}")
            
            train_losses = np.average(train_losses)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch: {epoch + 1} | Train Loss: {train_losses:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")

            # Log metrics for each epoch
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_losses,
                "validation_loss": vali_loss,
                "test_loss": test_loss,
            })
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            
        best_model_path = os.path.join(path, 'checkpoint.pth')

        # Save the best model to wandb
        upload_files_to_wandb(
            project_name=self.project_name,
            run_name=self.run_name,
            model_weights_path=best_model_path
        )

        # Save the last model to WandB
        final_model_artifact = wandb.Artifact('final_model_weights', type='model')
        final_model_artifact.add_file(best_model_path)
        wandb.log_artifact(final_model_artifact)

        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(vali_loader):
               
                batch_x, batch_y, batch_x_mark, batch_y_mark, site, batch_x_ts, batch_y_ts = data
               
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                site = site.to(self.device)
                batch_x_ts = batch_x_ts.to(self.device)
                batch_y_ts = batch_y_ts.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                mark = True if self.args.is_pretraining else False
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'LSTM' in self.args.model:
                            outputs = self.model(batch_x, mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'LSTM' in self.args.model:
                        outputs = self.model(batch_x, mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                
                if self.args.model != 'LSTM':
                    ### calculate metrics with only active power
                    output_np = outputs.detach().cpu().numpy() 
                    batch_y_np = batch_y.detach().cpu().numpy()
                    # print('output_np', output_np)
                    # print('batch_y', batch_y_np)
                    
                    # print(output_np.shape) (1024, 16, 1)
                    
                    # active_power_np = vali_data.inverse_transform(site[:, 0], output_np.copy())
                    # active_power_gt_np = vali_data.inverse_transform(site[:, 0], batch_y_np.copy())
                    active_power_np = vali_data.inverse_transform( output_np.copy())
                    active_power_gt_np = vali_data.inverse_transform(batch_y_np.copy())
                   

                    # # scaler적용 후, 다시 3d로 되돌리기
                    # active_power_np = active_power_np.reshape(output_np.shape[0], output_np.shape[1], -1)
                    # active_power_gt_np = active_power_gt_np.reshape(batch_y_np.shape[0], batch_y_np.shape[1], -1)


                    pred = torch.from_numpy(active_power_np).to(self.device)
                    gt = torch.from_numpy(active_power_gt_np).to(self.device)
                    # print('pred', pred)
                    # print('gt', gt)                
                else:
                    # TODO: LSTM일 때, 코드 수정 필요
                    pred_np = vali_data.inverse_transform(outputs.detach().cpu().numpy())
                    gt_np = vali_data.inverse_transform(batch_y.detach().cpu().numpy())

                    # pred_np = pred_np.reshape(-1, vali_data[-2], batch_y_np.shape[-1])
                    pred = torch.from_numpy(pred_np[:, :, -1])
                    gt = torch.from_numpy(gt_np[:, :, -1])

                loss = criterion(pred, gt) 
                total_loss.append(loss.item())
        
        self.model.train()
        total_losses = np.average(total_loss)
        return total_losses


    # test == 0: test when training is done
    # test == 1: test when --is_training is 0 in scripts
    def test(self, setting, model_path=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print(f'loading model: {model_path}')
            if model_path != None:
                self.model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/', setting, 'checkpoint.pth')))
            
        pred_list = []
        true_list = []
        pred_normalized_list = []
        true_normalized_list = []
        input_list = []

        folder_path = os.path.join('./test_results/', setting)
        # folder_path_inout = os.path.join('./test_results/', 'in+out', setting)
        # folder_path_out = os.path.join('./test_results/', 'out', setting)

        # if not os.path.exists(folder_path_inout):
        #     os.makedirs(folder_path_inout)
        # if not os.path.exists(folder_path_out):
        #     os.makedirs(folder_path_out)

        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for i, data in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, site, batch_x_ts, batch_y_ts = data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                site = site.to(self.device)
                batch_x_ts = batch_x_ts.to(self.device)
                batch_y_ts = batch_y_ts.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                mark = True if self.args.is_pretraining else False
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'LSTM' in self.args.model:
                            outputs = self.model(batch_x, mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'LSTM' in self.args.model:
                        outputs = self.model(batch_x, mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
               
                if self.args.model != 'LSTM':
                   
                    outputs_np = outputs.detach().cpu().numpy()
                    batch_y_np = batch_y.detach().cpu().numpy()

                    # de-normalize the data and prediction 
                    batch_x_np = batch_x.detach().cpu().numpy()
                    # batch_x_ap = test_data.inverse_transform(site[:, 0], batch_x_np.copy()[:, :, -1])   
                    # pred = test_data.inverse_transform(site[:, 0], outputs_np.copy())
                    # true = test_data.inverse_transform(site[:, 0], batch_y_np.copy())
                    batch_x_ap = test_data.inverse_transform(batch_x_np.copy()[:, :, -1])   
                    pred = test_data.inverse_transform(outputs_np.copy())
                    true = test_data.inverse_transform(batch_y_np.copy())


                    # normalized된 결과
                    pred_normalized = outputs_np
                    true_normalized = batch_y_np

                    # pred = pred.reshape(outputs_np.shape[0], outputs_np.shape[1], -1)
                    # true = true.reshape(batch_y_np.shape[0], batch_y_np.shape[1], -1)
                                 
                else:
                    # TODO: LSTM일 때, 코드 수정 필요
                    pred_np = test_data.inverse_transform(outputs.detach().cpu().numpy())
                    true_np = test_data.inverse_transform(batch_y.detach().cpu().numpy())
                    
                    # pred_np = pred_np.reshape(-1, outputs.shape[-2], batch_y_np.shape[-1])
                    # true_np = true_np.reshape(-1, outputs.shape[-2], batch_y_np.shape[-1])

                    pred = torch.from_numpy(pred_np)[:, :, -1]
                    true = torch.from_numpy(true_np)[:, :, -1]


                pred_list.append(pred)
                true_list.append(true[:,-self.args.pred_len:])
                pred_normalized_list.append(pred_normalized)
                true_normalized_list.append(true_normalized)
                input_list.append(batch_x.detach().cpu().numpy())
                
                # TODO: visualize code 수정 필요
                # Visualize periodically
                if i % 3 == 0:
                    self.plot_predictions(i, batch_x_ap[0], true[0], pred[0], folder_path)
                # if i % 10 == 0:
                #     if self.args.model != 'LSTM':
                #     # visualize_input_length = outputs.shape[1]*3 # visualize three times of the prediction length
                #         input_np_s = batch_x[:, :, -1].detach().cpu().numpy()
                #         input_inverse_transform_s = test_data.inverse_transform(input_np_s)
                #         input_seq_s = input_inverse_transform_s[0,:]
                #         gt_s = true[0, -self.args.pred_len:]
                #         pd_s = pred[0, :]
                #         visual(input_seq_s, gt_s, pd_s, os.path.join(folder_path_inout, str(i) + '.png'))
                #         # visual_out(input_seq, gt, pd, os.path.join(folder_path_out, str(i) + '.png'))
                #         # TODO: visual, visual_out은 거의 같은데 하나는 input을 포함하고 하나는 input을 포함하지 않는다.
                #     else:
                #         input_np = batch_x.detach().cpu().numpy()
                #         input_inverse_transform = test_data.inverse_transform(input_np)

                #         gt = np.concatenate((input_inverse_transform[0, :, -1], true[0, :, -1]), axis=0)
                #         pd = np.concatenate((input_inverse_transform[0, :, -1], pred[0, :, -1]), axis=0)
                #         visual_original(gt, pd, os.path.join(folder_path, str(i) + '.png'))
                    

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        pred_np = np.array(pred_list)
        trues_np = np.array(true_list)
        pred_normalized_np = np.array(pred_normalized_list)
        true_normalized_np = np.array(true_normalized_list)
        inputx_np = np.array(input_list)

        pred_np = pred_np.reshape(-1, pred_np.shape[-2], pred_np.shape[-1])
        trues_np = trues_np.reshape(-1, trues_np.shape[-2], trues_np.shape[-1])
        pred_normalized_np = pred_normalized_np.reshape(-1, pred_normalized_np.shape[-2], pred_normalized_np.shape[-1])
        true_normalized_np = true_normalized_np.reshape(-1, true_normalized_np.shape[-2], true_normalized_np.shape[-1])
        inputx_np = inputx_np.reshape(-1, inputx_np.shape[-2], inputx_np.shape[-1])

        # result save
        folder_path = os.path.join('./results/', setting)
        txt_save_path = os.path.join(folder_path,
                                     f"{self.args.seq_len}_{self.args.pred_len}_result.txt")
        txt_save_path_normalized = os.path.join(folder_path,
                                     f"{self.args.seq_len}_{self.args.pred_len}_result_normalized.txt")
        
        os.makedirs(folder_path, exist_ok=True)

        avg_mae, avg_mse, avg_rmse, avg_nrmse, avg_mape, avg_mspe, avg_rse, avg_r2 = metric(pred_np, trues_np, site_max_capacities=test_data.site_max_capacities, file_path=txt_save_path)
        avg_mae_n, avg_mse_n, avg_rmse_n, avg_nrmse_n, avg_mape_n, avg_mspe_n, avg_rse_n, avg_r2_n = metric(pred_normalized_np, true_normalized_np, site_max_capacities=test_data.site_max_capacities, file_path=txt_save_path_normalized)

        print('\navg_mae: {}, avg_mse: {}, avg_rmse: {}, avg_nrmse: {}, avg_mape: {}, avg_mspe: {}, avg_rse: {}, avg_r2: {}'.format(avg_mae, avg_mse, avg_rmse, avg_nrmse, avg_mape, avg_mspe, avg_rse, avg_r2))
        print('\navg_mae_normalized: {}, avg_mse_normalized: {}, avg_rmse_normalized: {}, avg_nrmse_normalized: {}, avg_mape_normalized: {}, avg_mspe_normalized: {}, avg_rse_normalized: {}, avg_r2_normalized: {}'.format(avg_mae_n, avg_mse_n, avg_rmse_n, avg_nrmse_n, avg_mape_n, avg_mspe_n, avg_rse_n, avg_r2_n))
        

        
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', pred_np)
        # np.save(folder_path + 'true.npy', trues_np)
        # np.save(folder_path + 'x.npy', inputx_np)
        return
    import matplotlib.pyplot as plt
    import os

    def plot_predictions(self, i, input_sequence, ground_truth, predictions, save_path):
        """
        예측 시각화 함수
        Args:
            input_sequence (numpy array): 입력 시퀀스 데이터
            ground_truth (numpy array): 실제값
            predictions (numpy array): 예측값
            save_path (str): 플롯을 저장할 경로
        """
        plt.figure(figsize=(12, 6))
        
        # 입력 시퀀스 플롯
        plt.plot(range(len(input_sequence)), input_sequence, label='Input Sequence', color='blue', linestyle='--')
        
        # 실제값 플롯
        prediction_start = len(input_sequence)
        plt.plot(range(prediction_start, prediction_start + len(ground_truth)), ground_truth, label='Ground Truth', color='green')
        
        # 예측값 플롯
        plt.plot(range(prediction_start, prediction_start + len(predictions)), predictions, label='Predictions', color='red')
        
        # 레이블, 제목 설정
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Prediction vs Ground Truth')
        plt.legend()
        
        # 플롯 저장
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'pred_{i}.png'))
        plt.close()        



    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        pred_list = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(pred_loader):

                batch_x, batch_y, batch_x_mark, batch_y_mark = data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                pred_list.append(pred)
                

        pred_np = np.array(pred_list)
        pred_np = pred_np.reshape(-1, pred_np.shape[-2], pred_np.shape[-1])

        # result save
        folder_path = os.path.join('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np_save_path = os.path.join(folder_path, "real_prediction_source.npy", pred_np)
        np.save(np_save_path)

        return
