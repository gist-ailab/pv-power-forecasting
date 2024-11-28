#!/bin/bash

DATE=$(date +%y%m%d%H)
model_name=PatchTST
model_id=$DATE
exp_id="${DATE}_Pretrain_German_$model_name"_individual_night

if [ ! -d "./logs/$exp_id" ]; then
    mkdir -p ./logs/$exp_id
fi

seq_len=256
pred_len=16
label_len=0

root_path_name="/ailab_mat/dataset/PV/Germany/processed_data_night/"
data_name=German
random_seed=2024

e_layers=4
n_heads=8
d_model=256
d_ff=512

export CUDA_VISIBLE_DEVICES=7
export SCRIPT_NAME=$(basename "$0" .sh)
for pred_len in 16 8 4 2 1
do
    python -u run_longExp.py \
        --checkpoints "${SCRIPT_NAME}_${seq_len}_${pred_len}" \
        --gpu 0 \
        --individual 1 \
        --random_seed $random_seed \
        --is_pretraining 1 \
        --root_path $root_path_name \
        --model_id $model_id \
        --model $model_name \
        --data $data_name \
        --features MS \
        --seq_len 256\
        --label_len $label_len \
        --pred_len $pred_len\
        --enc_in 5 \
        --e_layers $e_layers \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff $d_ff \
        --dropout 0.05\
        --fc_dropout 0.05\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --patience 20\
        --embed 'timeF' \
        --distributed \
        --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/$exp_id/$model_name'_'$data_name'_'$seq_len'_'$pred_len'_'$e_layers.log
done