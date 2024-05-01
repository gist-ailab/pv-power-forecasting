from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, PatchCDTST, LSTM
from models.Stat_models import Naive_repeat, Arima
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, visual_out, visual_original
from utils.metrics import metric
# from utils.mmd_loss import MMDLoss

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from pytorch_adapt.layers import MMDLoss, MMDBatchedLoss

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchCDTST': PatchCDTST,
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

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # cross_criterion = MMDLoss() # 안됨
        # cross_criterion = MMDBatchedLoss(batch_size=self.args.batch_size, mmd_type='quadratic')
        cross_criterion = MMDLoss(mmd_type="quadratic")
        flatten = nn.Flatten()
        # flatten은 [bs x nvars x d_model x patch_num]를 [bs x d_model x patch_num]을 nvars만큼 tensor를 갖는 list로 바꿔줄 때 사용
        # https://github.com/KevinMusgrave/pytorch-adapt/blob/main/src/pytorch_adapt/layers/mmd_loss.py
        if self.args.model == 'PatchCDTST' and self.args.is_training:
            return criterion, cross_criterion, flatten
        return criterion

    def train(self, setting, exp_id, resume):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        count = count_parameters(self.model)
        print(f'The number of parameters of the model is {count}.')        
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if 'checkpoint.pth' in self.args.checkpoints.split('/'):
            path = self.args.checkpoints
        else:
            path = os.path.join(self.args.checkpoints, exp_id, setting)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        
        if self.args.model == 'PatchCDTST':
            criterion, cross_criterion, flatten = self._select_criterion()
        else:
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
                
                if len(data) == 4: # Original data loader
                    batch_x_s, batch_y_s, batch_x_mark_s, batch_y_mark_s = data
                    batch_x_s = batch_x_s.float().to(self.device)
                    batch_y_s = batch_y_s.float().to(self.device)
                    batch_x_mark_s = batch_x_mark_s.float().to(self.device)
                    batch_y_mark_s = batch_y_mark_s.float().to(self.device)
                
                elif len(data) == 8: # Data loader for PatchCDTST
                    batch_x_s, batch_y_s, batch_x_mark_s, batch_y_mark_s, batch_x_t, batch_y_t, batch_x_mark_t, batch_y_mark_t = data   # s for source, t for target
                    batch_x_s = batch_x_s.float().to(self.device)
                    batch_y_s = batch_y_s.float().to(self.device)
                    batch_x_mark_s = batch_x_mark_s.float().to(self.device)
                    batch_y_mark_s = batch_y_mark_s.float().to(self.device)
                    batch_x_t = batch_x_t.float().to(self.device)
                    batch_y_t = batch_y_t.float().to(self.device)
                    batch_x_mark_t = batch_x_mark_t.float().to(self.device)
                    batch_y_mark_t = batch_y_mark_t.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y_s[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_s[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                            if 'CDTST' in self.args.model:
                                source_outputs, target_outputs, target_feat, cross_feat = self.model(batch_x_s, batch_x_t)
                            elif  'LSTM' in self.args.model:
                                source_outputs = self.model(batch_x_s, dec_inp)
                            else:
                                source_outputs = self.model(batch_x_s)
            
                        else:
                            if self.args.output_attention:
                                source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)[0]
                            else:
                                source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)

                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                        if 'CDTST' in self.args.model:
                            source_outputs, target_outputs, target_feat, cross_feat = self.model(batch_x_s, batch_x_t)
                        elif 'LSTM' in self.args.model:
                            source_outputs = self.model(batch_x_s, dec_inp)
                        else:
                            source_outputs = self.model(batch_x_s)
            
                    else:
                        if self.args.output_attention:
                            source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)[0]
                        else:
                            source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s, batch_y_s)
                            # TODO: batch_y_s 의 역할이 뭐지??? TST 계열에선 안 쓰긴 한다. train에만 들어가 있음.
                            # print(outputs.shape,batch_y.shape)
                            
                f_dim = -1 if self.args.features == 'MS' else 0
                
                # loss for source domain
                source_outputs = source_outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_s = batch_y_s[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_s = criterion(source_outputs, batch_y_s)
                # train_loss_s.append(loss_s.item())
                
                if 'CDTST' in self.args.model:
                    # loss for target domain
                    # target_outputs = target_outputs[:, -self.args.pred_len:, f_dim:]
                    # batch_y_t = batch_y_t[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # loss_t = criterion(target_outputs, batch_y_t)
                    t_outputs = target_outputs[:, -self.args.pred_len:, f_dim:]
                    b_y_t = batch_y_t[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_t = criterion(t_outputs, b_y_t)
                    
                    # loss for cross-domain                    
                    target_f = []
                    cross_f = []
                    # 차원 1은 nvars이고 이건 cross, target이 동일하여 동일 루프에서 사용
                    for i in range(target_feat.size(1)):
                        target_f_list = flatten(target_feat[:, i, :, :])
                        cross_f_list = flatten(cross_feat[:, i, :, :])
                        target_f.append(target_f_list)
                        cross_f.append(cross_f_list)
                    mmd_loss = cross_criterion(target_f, cross_f)   # list내에 각 변수 별로 tensor를 구성하여 MMD를 구하는 함수에 넣어 줌.

                    total_loss = loss_s + loss_t + mmd_loss
                else:
                    total_loss = loss_s
                    
                train_losses.append(total_loss.item())
                    

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {total_loss.item():.7f}, ")
                    if 'CDTST' in self.args.model:
                        print(f"loss_source: {loss_s.item():.7f}, loss_target: {loss_t.item():.7f}, mmd_loss: {mmd_loss.item():.7f} \n")
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                    
                else:
                    total_loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} | cost time: {time.time() - epoch_time}")
            
            train_losses = np.average(train_losses)
            if 'CDTST' in self.args.model:
                # vali_loss = self.vali(vali_data, vali_loader, criterion, cross_criterion, flatten)
                # test_loss = self.vali(test_data, test_loader, criterion, cross_criterion, flatten)
                val_total_loss, val_source_loss, val_target_loss = self.vali(vali_data, vali_loader, criterion, cross_criterion, flatten)
                test_total_loss, test_source_loss, test_target_loss = self.vali(test_data, test_loader, criterion, cross_criterion, flatten)
                print(f"Epoch: {epoch + 1} | Train Loss: {train_losses:.7f}")
                print(f"Vali Source Domain Loss: {val_source_loss:.7f}, Vali Target Domain Loss: {val_target_loss:.7f}")
                print(f"Test Source Domain Loss: {test_source_loss:.7f}, Test Target Domain Loss: {test_target_loss:.7f}")
                print(f"\n")
                vali_loss = val_target_loss
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                print(f"Epoch: {epoch + 1} | Train Loss: {train_losses:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion, cross_criterion=None, flatten=None):
        total_losses = []
        total_source_losses = []
        total_target_losses = []
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(vali_loader):
                if len(data) == 4: # Original data loader
                    batch_x_s, batch_y_s, batch_x_mark_s, batch_y_mark_s = data
                    batch_x_s = batch_x_s.float().to(self.device)
                    batch_y_s = batch_y_s.float().to(self.device)
                    batch_x_mark_s = batch_x_mark_s.float().to(self.device)
                    batch_y_mark_s = batch_y_mark_s.float().to(self.device)
                
                elif len(data) == 8: # Data loader for PatchCDTST
                    batch_x_s, batch_y_s, batch_x_mark_s, batch_y_mark_s, batch_x_t, batch_y_t, batch_x_mark_t, batch_y_mark_t = data   # s for source, t for target
                    batch_x_s = batch_x_s.float().to(self.device)
                    batch_y_s = batch_y_s.float().to(self.device)
                    batch_x_mark_s = batch_x_mark_s.float().to(self.device)
                    batch_y_mark_s = batch_y_mark_s.float().to(self.device)
                    batch_x_t = batch_x_t.float().to(self.device)
                    batch_y_t = batch_y_t.float().to(self.device)
                    batch_x_mark_t = batch_x_mark_t.float().to(self.device)
                    batch_y_mark_t = batch_y_mark_t.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y_s[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_s[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.mode or 'LSTM' in self.args.model:
                            if 'CDTST' in self.args.model:
                                source_outputs, target_outputs, target_feat, cross_feat = self.model(batch_x_s, batch_x_t)
                            else:
                                source_outputs = self.model(batch_x_s)
                        else:
                            if self.args.output_attention:
                                source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)[0]
                            else:
                                source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'LSTM' in self.args.model:
                        if 'CDTST' in self.args.model:
                            source_outputs, target_outputs, target_feat, cross_feat = self.model(batch_x_s, batch_x_t)
                        else:
                            source_outputs = self.model(batch_x_s)
                    else:
                        if self.args.output_attention:
                            source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)[0]
                        else:
                            source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)
                            
                f_dim = -1 if self.args.features == 'MS' else 0

                source_outputs = source_outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_s = batch_y_s[:, -self.args.pred_len:, f_dim:].to(self.device)
                                
                if self.args.model != 'LSTM':
                    ### calculate metrics with only active power
                    active_power_s = source_outputs[:, :, -1].detach().cpu()
                    active_power_gt_s = batch_y_s[:, :, -1].detach().cpu()

                    # de-normalize the data and prediction values
                    active_power_np_s = vali_data.inverse_transform(active_power_s)
                    active_power_gt_np_s = vali_data.inverse_transform(active_power_gt_s)
                
                    pred_s = torch.from_numpy(active_power_np_s).to(self.device)
                    gt_s = torch.from_numpy(active_power_gt_np_s).to(self.device)
                else:
                    pred_np = vali_data.inverse_transform(source_outputs.detach().cpu().numpy())
                    gt_np = vali_data.inverse_transform(batch_y_s.detach().cpu().numpy())

                    pred_s = torch.from_numpy(pred_np)
                    gt_s = torch.from_numpy(gt_np)

                loss_s = criterion(pred_s, gt_s)
                
                if 'CDTST' in self.args.model:
                    # loss for target domain
                    target_outputs = target_outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_t = batch_y_t[:, -self.args.pred_len:, f_dim:].to(self.device)
                    active_power_t = target_outputs[:, :, -1].detach().cpu()
                    active_power_gt_t = batch_y_t[:, :, -1].detach().cpu()

                    # de-normalize the data and prediction values for target domain
                    active_power_np_t = vali_data.inverse_transform(active_power_t, domain='target')
                    active_power_gt_np_t = vali_data.inverse_transform(active_power_gt_t, domain='target')
                    
                    pred_t = torch.from_numpy(active_power_np_t).to(self.device)
                    gt_t = torch.from_numpy(active_power_gt_np_t).to(self.device)
                    
                    loss_t = criterion(pred_t, gt_t)
                    
                    # loss for cross-domain                    
                    target_f = []
                    cross_f = []
                    # 차원 1은 nvars이고 이건 cross, target이 동일하여 동일 루프에서 사용
                    for i in range(target_feat.size(1)):
                        target_f_list = flatten(target_feat[:, i, :, :])
                        cross_f_list = flatten(cross_feat[:, i, :, :])
                        target_f.append(target_f_list)
                        cross_f.append(cross_f_list)
                    mmd_loss = cross_criterion(target_f, cross_f)   # list내에 각 변수 별로 tensor를 구성하여 MMD를 구하는 함수에 넣어 줌.
                    
                    total_loss = loss_s + loss_t + mmd_loss
                    
                else:
                    total_loss = loss_s
                    
                
                total_target_losses.append(loss_t.item())
                if 'CDTST' in self.args.model:
                    # train loop에서 early stopping을 위해 source, target loss를 따로 저장
                    total_losses.append(total_loss.item())
                    total_source_losses.append(loss_s.item())
        
        self.model.train()
        if 'CDTST' in self.args.model:
            total_losses = np.average(total_losses)
            total_source_losses = np.average(total_source_losses)
            total_target_losses = np.average(total_target_losses)
            # print(f"Vali Loss: {total_losses:.7f}, Source Loss: {total_source_losses:.7f}, Target Loss: {total_target_losses:.7f}")
            # CDTST는 early stopping 적용을 target loss를 통해 확인하며, source loss, total loss는 확인용
            return total_losses, total_source_losses, total_target_losses
        else:
            total_losses = np.average(total_losses)
            return total_losses


    def test(self, setting, exp_id, model_path=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        # pv_max = test_loader.sampler.data_source.pv_max
        # pv_min = test_loader.sampler.data_source.pv_min
        
        if test:
            print('loading model')
            if model_path != None:
                self.model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/', f'exp{exp_id}', setting, 'checkpoint.pth')))
            
        preds_s = []
        trues_s = []
        inputx_s = []
        preds_t = []
        trues_t = []
        inputx_t = []
        
        folder_path = './test_results/' + exp_id + '/' + setting + '/'
        folder_path_inout = './test_results/' + exp_id + '/in+out/' + setting + '/'
        folder_path_out = './test_results/' + exp_id + '/out/' + setting + '/'
        if not os.path.exists(folder_path_inout):
            os.makedirs(folder_path_inout)
        if not os.path.exists(folder_path_out):
            os.makedirs(folder_path_out)

        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for i, data in enumerate(test_loader):
                if len(data) == 4: # Original data loader
                    batch_x_s, batch_y_s, batch_x_mark_s, batch_y_mark_s = data
                    batch_x_s = batch_x_s.float().to(self.device)
                    batch_y_s = batch_y_s.float().to(self.device)
                    batch_x_mark_s = batch_x_mark_s.float().to(self.device)
                    batch_y_mark_s = batch_y_mark_s.float().to(self.device)
                
                elif len(data) == 8: # Data loader for PatchCDTST
                    batch_x_s, batch_y_s, batch_x_mark_s, batch_y_mark_s, batch_x_t, batch_y_t, batch_x_mark_t, batch_y_mark_t = data   # s for source, t for target
                    batch_x_s = batch_x_s.float().to(self.device)
                    batch_y_s = batch_y_s.float().to(self.device)
                    batch_x_mark_s = batch_x_mark_s.float().to(self.device)
                    batch_y_mark_s = batch_y_mark_s.float().to(self.device)
                    batch_x_t = batch_x_t.float().to(self.device)
                    batch_y_t = batch_y_t.float().to(self.device)
                    batch_x_mark_t = batch_x_mark_t.float().to(self.device)
                    batch_y_mark_t = batch_y_mark_t.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y_s[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_s[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'LSTM' in self.args.model:
                            if 'CDTST' in self.args.model:
                                source_outputs, target_outputs, target_feat, cross_feat = self.model(batch_x_s, batch_x_t)
                            else:
                                source_outputs = self.model(batch_x_s)
                        else:
                            if self.args.output_attention:
                                source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)[0]
                            else:
                                source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'LSTM' in self.args.model:
                        if 'CDTST' in self.args.model:
                            source_outputs, target_outputs, target_feat, cross_feat = self.model(batch_x_s, batch_x_t)
                        else:
                            source_outputs = self.model(batch_x_s)
                    else:
                        if self.args.output_attention:
                            source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)[0]
                        else:
                            source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)

                source_outputs = source_outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_s = batch_y_s[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # source domain
                if self.args.model != 'LSTM':
                    ### calculate metrics with only active power
                    active_power_s = source_outputs[:, :, -1].detach().cpu()
                    active_power_gt_s = batch_y_s[:, :, -1].detach().cpu()
                    
                    # de-normalize the data and prediction values
                    pred_s = test_data.inverse_transform(active_power_s)
                    true_s = test_data.inverse_transform(active_power_gt_s)
               
                else:
                    pred_np = test_data.inverse_transform(source_outputs.detach().cpu().numpy())
                    true_np = test_data.inverse_transform(batch_y_s.detach().cpu().numpy())

                    pred_s = torch.from_numpy(pred_np)
                    true_s = torch.from_numpy(true_np)

                preds_s.append(pred_s)
                trues_s.append(true_s[:,-self.args.pred_len:])
                inputx_s.append(batch_x_s.detach().cpu().numpy())
                
                if 'CDTST' not in self.args.model and i % 10 == 0:
                    if self.args.model != 'LSTM':
                    # visualize_input_length = outputs.shape[1]*3 # visualize three times of the prediction length
                        input_np_s = batch_x_s[:, :, -1].detach().cpu().numpy()
                        input_inverse_transform_s = test_data.inverse_transform(input_np_s)
                        input_seq_s = input_inverse_transform_s[0,:]
                        gt_s = true_s[0, -self.args.pred_len:]
                        pd_s = pred_s[0, :]
                        visual(input_seq_s, gt_s, pd_s, os.path.join(folder_path_inout, str(i) + '.png'))
                        # visual_out(input_seq, gt, pd, os.path.join(folder_path_out, str(i) + '.png'))
                        # TODO: visual, visual_out은 거의 같은데 하나는 input을 포함하고 하나는 input을 포함하지 않는다.
                    else:
                        input_np = batch_x_s.detach().cpu().numpy()
                        input_inverse_transform = test_data.inverse_transform(input_np)

                        gt = np.concatenate((input_inverse_transform[0, :, -1], true_s[0, :, -1]), axis=0)
                        pd = np.concatenate((input_inverse_transform[0, :, -1], pred_s[0, :, -1]), axis=0)
                        visual_original(gt, pd, os.path.join(folder_path, str(i) + '.png'))
                
                if 'CDTST' in self.args.model and i % 10 == 0:
                    # target domain
                    target_outputs = target_outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y_t = batch_y_t[:, -self.args.pred_len:, f_dim:].to(self.device)
                    active_power_t = target_outputs[:, :, -1].detach().cpu()
                    active_power_gt_t = batch_y_t[:, :, -1].detach().cpu()
                    
                    # de-normalize the data and prediction values
                    pred_t = test_data.inverse_transform(active_power_t, domain='target')
                    true_t = test_data.inverse_transform(active_power_gt_t, domain='target')

                    preds_t.append(pred_t)
                    trues_t.append(true_t)
                    inputx_t.append(batch_x_t.detach().cpu().numpy())
                    if i % 10 == 0:                    
                        input_np_t = batch_x_t[:, :, -1].detach().cpu().numpy()
                        input_inverse_transform_t = test_data.inverse_transform(input_np_t, domain='target')
                        input_seq_t = input_inverse_transform_t[0,:]
                        gt_t = true_t[0, -self.args.pred_len:]
                        pd_t = pred_t[0, :]
                        visual(input_seq_t, gt_t, pd_t, os.path.join(folder_path_inout, str(i) + '.png'))
                    
                    

        if self.args.test_flop:
            test_params_flop((batch_x_s.shape[1],batch_x_s.shape[2]))
            if 'CDTST' in self.args.model:
                test_params_flop((batch_x_t.shape[1],batch_x_t.shape[2]))
            exit()

        preds_s = np.array(preds_s)
        trues_s = np.array(trues_s)
        inputx_s = np.array(inputx_s)
        if 'CDTST' in self.args.model:
            preds_t = np.array(preds_t)
            trues_t = np.array(trues_t)
            inputx_t = np.array(inputx_t)

        preds_s = preds_s.reshape(-1, preds_s.shape[-2], preds_s.shape[-1])
        trues_s = trues_s.reshape(-1, trues_s.shape[-2], trues_s.shape[-1])
        inputx_s = inputx_s.reshape(-1, inputx_s.shape[-2], inputx_s.shape[-1])
        if 'CDTST' in self.args.model:
            preds_t = preds_t.reshape(-1, preds_t.shape[-2], preds_t.shape[-1])
            trues_t = trues_t.reshape(-1, trues_t.shape[-2], trues_t.shape[-1])
            inputx_t = inputx_t.reshape(-1, inputx_t.shape[-2], inputx_t.shape[-1])

        # result save
        folder_path = './results/' + exp_id+ '/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        # mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        
        # calculate metrics with only generated power
        
        mae_s, mse_s, rmse_s = metric(preds_s, trues_s)
        print('mse_source:{}, mae_source:{}, rmse_source:{}'.format(mse_s, mae_s, rmse_s))
        if 'CDTST' in self.args.model:
            mae_t, mse_t, rmse_t = metric(preds_t, trues_t)
            print('mse_target:{}, mae_target:{}, rmse_target:{}'.format(mse_t, mae_t, rmse_t))
        
        txt_save_path = os.path.join(folder_path,
                                     f"{self.args.seq_len}_{self.args.pred_len}_result.txt")
        f = open(txt_save_path, 'a')
        f.write(exp_id + "  \n")
        f.write(setting + "  \n")
        f.write('mse_source:{}, mae_source:{}, rmse_source:{}'.format(mse_s, mae_s, rmse_s))
        if 'CDTST' in self.args.model:
            f.write('mse_target:{}, mae_target:{}, rmse_target:{}'.format(mse_t, mae_t, rmse_t))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred_source.npy', preds_s)
        # np.save(folder_path + 'true_source.npy', trues_s)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, exp_id, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds_s = []
        preds_t = []        

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(pred_loader):
                if len(data) == 4: # Original data loader
                    batch_x_s, batch_y_s, batch_x_mark_s, batch_y_mark_s = data
                    batch_x_s = batch_x_s.float().to(self.device)
                    batch_y_s = batch_y_s.float().to(self.device)
                    batch_x_mark_s = batch_x_mark_s.float().to(self.device)
                    batch_y_mark_s = batch_y_mark_s.float().to(self.device)
                
                elif len(data) == 8: # Data loader for PatchCDTST
                    batch_x_s, batch_y_s, batch_x_mark_s, batch_y_mark_s, batch_x_t, batch_y_t, batch_x_mark_t, batch_y_mark_t = data   # s for source, t for target
                    batch_x_s = batch_x_s.float().to(self.device)
                    batch_y_s = batch_y_s.float().to(self.device)
                    batch_x_mark_s = batch_x_mark_s.float().to(self.device)
                    batch_y_mark_s = batch_y_mark_s.float().to(self.device)
                    batch_x_t = batch_x_t.float().to(self.device)
                    batch_y_t = batch_y_t.float().to(self.device)
                    batch_x_mark_t = batch_x_mark_t.float().to(self.device)
                    batch_y_mark_t = batch_y_mark_t.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y_s[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_s[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            if 'CDTST' in self.args.model:
                                source_outputs, target_outputs, target_feat, cross_feat = self.model(batch_x_s, batch_x_t)
                            else:
                                source_outputs = self.model(batch_x_s)
                        else:
                            if self.args.output_attention:
                                source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)[0]
                            else:
                                source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        if 'CDTST' in self.args.model:
                            source_outputs, target_outputs, target_feat, cross_feat = self.model(batch_x_s, batch_x_t)
                        else:
                            source_outputs = self.model(batch_x_s)
                    else:
                        if self.args.output_attention:
                            source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)[0]
                        else:
                            source_outputs = self.model(batch_x_s, batch_x_mark_s, dec_inp, batch_y_mark_s)
                
                pred_s = source_outputs.detach().cpu().numpy()  # .squeeze()
                preds_s.append(pred_s)
                if 'CDTST' in self.args.model:
                    pred_t = target_outputs.detach().cpu().numpy()  # .squeeze()
                    preds_t.append(pred_t)
                

        preds_s = np.array(preds_s)
        preds_s = preds_s.reshape(-1, preds_s.shape[-2], preds_s.shape[-1])

        # result save
        folder_path = './results/' + exp_id + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction_source.npy', preds_s)
        
        if 'CDTST' in self.args.model:
            preds_t = np.array(preds_t)
            preds_t = preds_t.reshape(-1, preds_t.shape[-2], preds_t.shape[-1])
            np.save(folder_path + 'real_prediction_target.npy', preds_t)

        return
