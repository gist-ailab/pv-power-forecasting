from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, LSTM
from models.Stat_models import Naive_repeat, Arima
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, visual_out, visual_original
from utils.metrics import MetricEvaluator

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os

import os
import matplotlib.pyplot as plt
import pandas as pd

import os
import time
import datetime

import warnings
import matplotlib.pyplot as plt
import numpy as np
import wandb
from utils.wandb_uploader import upload_files_to_wandb
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Exp_Freeze(Exp_Basic):
    def __init__(self, args):
        super(Exp_Freeze, self).__init__(args)
        self.project_name = "pv-forecasting-freeze-test"
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
        
        model = model_dict[self.args.model].Model(self.args).float()
        
        # 먼저 모델을 GPU로 이동
        model = model.to(self.device)
        
        if self.args.distributed:
        # 그 다음 DistributedDataParallel 설정
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )

        if self.args.resume:
            model = self.load_model(model, self.args.checkpoints)

        if self.args.num_freeze_layers > 0:
            model = self.load_model(model, self.args.checkpoints)
            
            # Define the layers to freeze
            freeze_layers = ['W_pos', 'W_P.weight', 'W_P.bias']
            freeze_layers += [f'model.backbone.encoder.layers.{i}' for i in range(self.args.num_freeze_layers)]
            
            # Freeze the specified layers
            for name, param in model.named_parameters():
                # freeze_layers에 해당하는 레이어 이름이 포함된 파라미터는 requires_grad=False로 설정
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False

            # Check which layers are frozen
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(f"Layer {name} is frozen.")
                    print(f"sdp_attn (scaled dot-product attention) layer is like a constant, so it is already frozen.")
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.args.distributed)
        return data_set, data_loader

    def _select_optimizer(self, part=None):
        if part is None:
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def masked_loss(self, predictions, targets, mask_value=-9999, loss_fn=torch.nn.MSELoss()):
        """
        Custom loss function to ignore specific mask_value during loss calculation.
        :param predictions: Model predictions [batch_size, seq_len, num_features]
        :param targets: Ground truth values [batch_size, seq_len, num_features]
        :param mask_value: Value to ignore in loss calculation
        :param loss_fn: Base loss function (e.g., MSELoss, MAELoss)
        """
        mask = (targets != mask_value)  # True for valid data
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]
        return loss_fn(valid_predictions, valid_targets)
    
    def load_model(self, model, checkpoints_path):
        latest_model_path = os.path.join(checkpoints_path, 'model_latest.pth')
        model.load_state_dict(torch.load(latest_model_path))
        print(f'Model loaded from {latest_model_path}')
        return model

    def train(self, checkpoints):
        self.args.checkpoints = os.path.join('checkpoints', checkpoints)
        # wandb 관련 작업은 rank 0에서만 실행
        if (self.args.local_rank == 0) and self.args.wandb:
            self._set_wandb(checkpoints)
            config = {
                "model": self.args.model,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
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
                "num_freeze_layers": self.args.num_freeze_layers,
            }
            upload_files_to_wandb(
                project_name=self.project_name,
                run_name=self.run_name,
                config=config
            )        
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints) if 'checkpoint.pth' not in self.args.checkpoints else self.args.checkpoints
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_losses = []
            epoch_time = time.time()
            
            self.model.train()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, site, batch_x_ts, batch_y_ts) in enumerate(train_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                pretrain_flag = True if self.args.is_pretraining else False
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                            outputs = self.model(batch_x, pretrain_flag)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                        outputs = self.model(batch_x, pretrain_flag)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    outputs = outputs[:, -self.args.pred_len:, -1:]
                    batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    # loss = self.masked_loss(outputs, batch_y, mask_value=-9999, loss_fn=criterion)  ### BSH
                    
                    loss.backward()
                    model_optim.step()
                
                train_losses.append(loss.item())

                if self.args.local_rank == 0 and (i + 1) % 100 == 0:
                    if self.args.wandb:
                        wandb.log({
                            "iteration": (epoch * len(train_loader)) + i + 1,
                            "train_loss_iteration": loss.item()
                        })
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - epoch_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    epoch_time = time.time()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            train_loss = np.average(train_losses)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            
            if self.args.local_rank == 0:
                print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")
                print(f"└ cost time: {time.time() - epoch_time}")
                if self.args.wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "validation_loss": vali_loss,
                        "test_loss": test_loss,
                    })
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f'Learning rate updated to {scheduler.get_last_lr()[0]}')
        
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if self.args.wandb == 0:
            upload_files_to_wandb(
                project_name=self.project_name,
                run_name=self.run_name,
                model_weights_path=best_model_path
            )

        final_model_artifact = wandb.Artifact('final_model_weights', type='model')
        final_model_artifact.add_file(best_model_path)
        wandb.log_artifact(final_model_artifact)

        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, site, batch_x_ts, batch_y_ts) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                pretrain_flag = True if self.args.is_pretraining else False
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                            outputs = self.model(batch_x, pretrain_flag)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                        outputs = self.model(batch_x, pretrain_flag)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        
        self.model.train()
        return np.average(total_loss)

    def test(self, model_path=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        folder_path = os.path.join('./test_results/', model_path)
        os.makedirs(folder_path, exist_ok=True)

        if 'checkpoint.pth' not in model_path:
            model_path = os.path.join(f'{self.args.checkpoints}', 'checkpoint.pth')
        
        self.model.load_state_dict(torch.load(model_path))
        
        evaluator = MetricEvaluator(file_path=os.path.join(folder_path, "site_metrics.txt"))
        scale_groups = evaluator.generate_scale_groups_for_dataset(self.args.data)

        pred_list = []
        true_list = []
        input_list = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, installation, batch_x_ts, batch_y_ts) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                installation = installation.to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                pretrain_flag = True if self.args.is_pretraining else False
              
                if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                    outputs = self.model(batch_x, pretrain_flag)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()
                batch_x_np = batch_x.detach().cpu().numpy()
                
                input_seq = test_data.inverse_transform(installation[:, 0], batch_x_np.copy())
                pred = test_data.inverse_transform(installation[:, 0], outputs_np.copy())
                true = test_data.inverse_transform(installation[:, 0], batch_y_np.copy())
                # print(pred.max(), pred.min(), flush=True)
                # print(true.max(), true.min(), flush=True)

                # # 예측값 범위 로깅
                # if i % 100 == 0:
                #     print("\n" + "="*50)
                #     print(f"Batch {i} - Prediction Range: [{pred.min():.4f}, {pred.max():.4f}]", flush=True)
                #     print(f"Batch {i} - True Range: [{true.min():.4f}, {true.max():.4f}]", flush=True)
                #     print("="*50 + "\n")

                # denormalized 데이터로 평가 수행
                # evaluator.update(preds=outputs_np, targets=batch_y_np)
                evaluator.update(preds=pred, targets=true)
                
                # pred_list.append(outputs_np)
                # true_list.append(batch_y_np)
                # input_list.append(batch_x_np)
                pred_list.append(pred)
                true_list.append(true)
                input_list.append(input_seq)

                if i % 2 == 0:
                    # self.plot_predictions(i, batch_x_np[0, -5:, -1], batch_y_np[0], outputs_np[0], folder_path)
                    self.plot_predictions(i, input_seq[0, -5:, -1], true[0], pred[0], folder_path)


                

                
                # # wandb에도 로깅
                # wandb.log({
                #     f"test/batch_{i}/pred_max": pred.max(),
                #     f"test/batch_{i}/pred_min": pred.min(),
                #     f"test/batch_{i}/true_max": true.max(),
                #     f"test/batch_{i}/true_min": true.min()
                # })
        print(f"Plotting complete. Results saved in {folder_path}")
        # results = evaluator.evaluate(scale_groups)
        results = evaluator.evaluate_scale_metrics(scale_groups)
        results_installation_mape = evaluator.evaluate_installation_metrics()
        for scale_name, metrics in results:
            rmse, mae, mbe, r2 = metrics
            print(f'Scale: {scale_name}')
            print(f'RMSE: {rmse:.4f}')
            print(f'MAE: {mae:.4f}')
            print(f'MAPE: {results_installation_mape:.4f}')
            print(f'MBE: {mbe:.4f}')
            print(f'R2: {r2:.4f}')


            # rmse, nrmse_range, nrmse_mean, mae, nmae, mape, mbe, r2 = metrics
            # print(f'Scale: {scale_name}')
            # print(f'RMSE: {rmse:.4f}')
            # print(f'NRMSE (Range): {nrmse_range:.4f}')
            # print(f'NRMSE (Mean): {nrmse_mean:.4f}')
            # print(f'MAE: {mae:.4f}')
            # print(f'NMAE: {nmae:.4f}')
            # print(f'MAPE: {mape:.4f}')
            # print(f'MBE: {mbe:.4f}')
            # print(f'R2: {r2:.4f}')


    def plot_predictions(self, i, input_sequence, ground_truth, predictions, save_path):
        """
        예측 시각화 함수 (인덱스 기반, 시각적 개선)
        Args:
            input_sequence (numpy array): 입력 시퀀스 데이터
            ground_truth (numpy array): 실제값
            predictions (numpy array): 예측값
            save_path (str): 플롯을 저장할 경로
        """
        # 인덱스 기반으로 x축을 설정
        input_index = np.arange(len(input_sequence))
        
        # ground_truth와 predictions에 input_sequence의 마지막 값을 앞에 추가하여 연결
        ground_truth = np.insert(ground_truth, 0, input_sequence[-1])
        predictions = np.insert(predictions, 0, input_sequence[-1])
        
        ground_truth_index = np.arange(len(input_sequence) - 1, len(input_sequence) + len(ground_truth) - 1)
        predictions_index = np.arange(len(input_sequence) - 1, len(input_sequence) + len(predictions) - 1)

        plt.figure(figsize=(14, 8))  # 더 큰 크기로 설정하여 가독성 향상

        # 입력 시퀀스의 마지막 5개 데이터만 플롯 (점선과 작은 점 추���, 투명도 적용)
        plt.plot(input_index[-10:], input_sequence.squeeze()[-10:], label='Input Sequence', color='royalblue', linestyle='--', alpha=0.7)
        plt.scatter(input_index[-10:], input_sequence.squeeze()[-10:], color='royalblue', s=10, alpha=0.6)

        # 수정된 ground_truth 사용하여 실제값 플롯 (굵기와 투명도 적용)
        plt.plot(ground_truth_index, ground_truth.squeeze(), label='Ground Truth', color='green', linewidth=2, alpha=0.8)
        
        # 예측값 플롯 (굵기와 투명도 적용)
        plt.plot(predictions_index, predictions.squeeze(), label='Predictions', color='red', linewidth=2, alpha=0.8)

        # 레이블, 제목 설정
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Prediction vs Ground Truth', fontsize=14)
        
        # 레전드를 오른쪽 상단에 고정
        plt.legend(loc='upper right', fontsize=10)
        
        # Grid 추가
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # 플롯 저장
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'pred_{i}.png'))
        plt.close()    # def plot_predictions(self, i, input_sequence, ground_truth, predictions, save_path):
    #     """
    #     예측 시각화 함수 (인덱스 기반)
    #     Args:
    #         input_sequence (numpy array): 입력 시퀀스 데이터
    #         ground_truth (numpy array): 실제값
    #         predictions (numpy array): 예측값
    #         save_path (str): 플롯을 저장할 경로
    #     """
    #     # 인덱스 기반으로 x축을 설정
    #     input_index = np.arange(len(input_sequence))
        
    #     # ground_truth와 predictions에 input_sequence의 마지막 값을 앞에 추가하여 연결
    #     ground_truth = np.insert(ground_truth, 0, input_sequence[-1])
    #     predictions = np.insert(predictions, 0, input_sequence[-1])
    #     ground_truth_index = np.arange(len(input_sequence) - 1, len(input_sequence) + len(ground_truth) - 1)
        
    #     predictions_index = np.arange(len(input_sequence) -1, len(input_sequence) + len(predictions)-1)

    #     plt.figure(figsize=(12, 6))

    #     # 입력 시퀀스 플롯
    #     plt.plot(input_index, input_sequence.squeeze(), label='Input Sequence', color='blue', linestyle='--')
        
    #     # 수정된 ground_truth 사용하여 실제값 플롯
    #     plt.plot(ground_truth_index, ground_truth.squeeze(), label='Ground Truth', color='green')
        
    #     # 예측값 플롯
    #     plt.plot(predictions_index, predictions.squeeze(), label='Predictions', color='red')

    #     # 레이블, 제목 설정
    #     plt.xlabel('Index')
    #     plt.ylabel('Value')
    #     plt.title('Prediction vs Ground Truth')
        
    #     # 레전드를 오른쪽 상단에 고정
    #     plt.legend(loc='upper right')

    #     # 플롯 저장
    #     os.makedirs(save_path, exist_ok=True)
    #     plt.savefig(os.path.join(save_path, f'pred_{i}.png'))
    #     plt.close()   
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
