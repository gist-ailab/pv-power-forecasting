import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import argparse
import torch

from models.PatchTST import Model
from utils.tools import StoreDictKeyPair
from data_provider.data_factory import data_provider


def _acquire_device(args):
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            args.gpu) if not args.use_multi_gpu else args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device


if __name__ == '__main__':
    wandb.login(key='59b6335c1210c476af746542d2e4768c161a712c')
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='PatchTST',
                        help='model name, options: [Autoformer, Informer, Transformer, DLinear, NLinear, Linear, PatchTST, PatchCDTST, Naive_repeat, Arima]')
    # data loader
    parser.add_argument('--data', type=str, default='DKASC_AliceSprings', help='dataset type. ex: DKASC, GIST')
    parser.add_argument('--root_path', type=str, default='/PV/DKASC_AliceSprings/converted', help='root path of the source domain data file')
    parser.add_argument('--data_path', type=str, action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", default={"type":"debug","train":"79-Site_DKA-M6_A-Phase.csv","val":"100-Site_DKA-M1_A-Phase.csv","test":"85-Site_DKA-M7_A-Phase.csv"},
                        help='In Debuggig, "type=debug,train=79-Site_DKA-M6_A-Phase.csv,val=100-Site_DKA-M1_A-Phase.csv,test=85-Site_DKA-M7_A-Phase.csv"')
    # parser.add_argument('--root_path', type=str, default='./data/GIST_dataset/', help='root path of the source domain data file')
    # parser.add_argument('--data_path', type=str, default='GIST_sisuldong.csv', help='source domain data file')
    parser.add_argument('--scaler', type=str, default='StandardScaler', help='StandardScaler, MinMaxScaler')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Active_Power', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=256, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length') # decoder 있는 모델에서 사용
    parser.add_argument('--pred_len', type=int, default=16, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=5, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    
    #LSTM
    parser.add_argument('--input_dim', type=int, default=5, help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--bidirectional', type=bool, default=True, help='bidirectional')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    parser.add_argument('--resume', action='store_true', default=False, help='resume')

    args = parser.parse_args()
   

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)


# WandB 설정
wandb.init(project="inference-visualization", name="inference_plot-average")
device = _acquire_device(args)
model = Model(args).to(device)
model.load_state_dict(torch.load('/home/pv/code/PatchTST/checkpoints/24110202_PatchTST_DKASC_AliceSprings_ftMS_sl256_ll0_pl16_dm256_nh8_el4_dl1_df512_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth'))

dataset, dataloader = data_provider(args, 'pred')
import torch





# 각 위치별 평균 계산
# Initialize arrays
pred_sum = torch.zeros(50).to(device)
pred_cnt = torch.zeros(50).to(device)  # 0으로 초기화
y_list = torch.zeros(50).to(device)
y_ts_list = [None] * 50

for i, data in enumerate(dataloader):
    x, y, x_mark, y_mark, site_idx, x_ts, y_ts = data
    x = x.float().to(device)
    y = y.float().to(device)
    x_mark = x_mark.to(device)
    y_mark = y_mark.to(device)
    site_idx = site_idx.to(device)
    x_ts = x_ts.to(device)
    y_ts = y_ts.to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(x)
        outputs = outputs.squeeze()

    
        # Get lengths
        output_length = outputs.shape[0]
        y_squeezed = y.squeeze()
        y_length = y_squeezed.shape[0]
        

        max_length = min(len(pred_sum) - i, output_length, y_length)

        # Update sums and counts
        pred_sum[i: i + max_length] += outputs[:max_length][:, -1]
        y_list[i: i + max_length] = y_squeezed[:max_length]
        pred_cnt[i: i + max_length] += 1  # pred_cnt 업데이트

        # Process timestamps
        y_ts_processed = [int(''.join(map(str, row.tolist()))) for row in y_ts.squeeze().cpu().numpy()]
        y_ts_list[i: i + max_length] = y_ts_processed[:max_length]

        if i + max_length >= 50:
            break

# Calculate mean prediction
mean_pred = pred_sum / pred_cnt
mean_pred = mean_pred.cpu().numpy()
y_list = y_list.cpu().numpy()

# Print and log results
for idx in range(len(mean_pred)):
    print(mean_pred[idx], y_list[idx], y_ts_list[idx])

    # Log to WandB
    wandb.log({
        "Predicted Values": mean_pred[idx],
        "True Values": y_list[idx],
        "Y Timestamp": y_ts_list[idx]
    })

wandb.finish()
