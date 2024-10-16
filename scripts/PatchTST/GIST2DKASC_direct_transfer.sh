#!/bin/bash

DATE=$(date +%y%m%d%H)
model_name=PatchTST
model_id=$DATE
exp_id="${DATE}_Direct_Transfer_GIST2DKASC_$model_name"

if [ ! -d "./logs/$exp_id" ]; then
    mkdir -p ./logs/$exp_id
fi

seq_len=336
label_len=0

root_path_name=/PV/DKASC_AliceSprings_1h
data_path_name=ALL
data_name=DKASC
random_seed=2024




for pred_len in 24 #1 2 4 8 16
do
    python -u run_longExp.py \
      --gpu 0 \
      --use_amp \
      --random_seed $random_seed \
      --is_training 0 \
      --checkpoints /home/pv/code/PatchTST/checkpoints/24100802_PatchTST_GIST_ftMS_sl336_ll0_pl24_dm128_nh16_el5_dl1_df1024_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth\
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id \
      --model $model_name \
      --data $data_name \
      --features MS \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 5 \
      --e_layers 5 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 1024 \
      --dropout 0.05\
      --fc_dropout 0.05\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --embed 'timeF' \
      --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/$exp_id/$model_name'_'$data_name'_'$seq_len'_'$pred_len.log
done