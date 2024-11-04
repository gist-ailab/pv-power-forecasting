#!/bin/bash

DATE=$(date +%y%m%d%H)
model_name=PatchTST
model_id=$DATE
exp_id="${DATE}_Fully_Finetune_infer_DKASC2UK_$model_name"

if [ ! -d "./logs/$exp_id" ]; then
    mkdir -p ./logs/$exp_id
fi

seq_len=256
label_len=0

root_path_name=/home/seongho_bak/Projects/PatchTST/data/UK_data/preprocessed
data_path_name='type=all'
data_name=UK
random_seed=2024


export CUDA_VISIBLE_DEVICES=1

for pred_len in 16 #1 2 4 8 16
do
    python -u run_longExp.py \
      --gpu 0 \
      --use_amp \
      --individual 1 \
      --random_seed $random_seed \
      --is_inference 1 \
      --checkpoints  /home/seongho_bak/Projects/PatchTST/checkpoints/fully_finetune/fully_finetune_24110407_PatchTST_UK_ftMS_sl256_ll0_pl16_dm256_nh8_el4_dl1_df512_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth\
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
      --e_layers 4 \
      --n_heads 8 \
      --d_model 256 \
      --d_ff 512 \
      --dropout 0.05\
      --fc_dropout 0.05\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --embed 'timeF' \
      --itr 1 --batch_size 1024 --learning_rate 0.0001 >logs/$exp_id/$model_name'_'$data_name'_'$seq_len'_'$pred_len.log
done