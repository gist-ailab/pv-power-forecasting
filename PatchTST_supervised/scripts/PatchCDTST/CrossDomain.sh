#!/bin/bash

DATE=$(date +%y%m%d%H)
model_name=PatchCDTST
exp_id="${DATE}_FirstSolar2GIST_$model_name"

if [ ! -d "./logs/$exp_id" ]; then
    mkdir ./logs/$exp_id
fi

seq_len=336

source_root_path=./dataset/DKASC/
target_root_path=./dataset/GIST_dataset/
source_data_path_name='79-Site_DKA-M6_A-Phase.csv'
target_data_path_name='GIST_sisuldong.csv'
data_name=CrossDomain

random_seed=2021

for pred_len in 1 2 4 8 16
do
    if [ $pred_len -eq 1 ]; then
        label_len=0
    else
        label_len=$((pred_len/2))
    fi
    python -u run_longExp.py \
      --gpu 0 \
      --random_seed $random_seed \
      --is_training 1 \
      --source_root_path $source_root_path \
      --target_root_path $target_root_path \
      --source_data_path $source_data_path_name \
      --target_data_path $target_data_path_name \
      --model_id $exp_id'_'$seq_len'_'$pred_len \
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
      --d_ff 512 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --embed 'timeF' \
      --exp_id $exp_id \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/$exp_id/$exp_id'_'$seq_len'_'$pred_len.log 
done