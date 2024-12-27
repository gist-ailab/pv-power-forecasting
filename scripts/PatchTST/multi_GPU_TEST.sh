#!/bin/bash

DATE=$(date +%y%m%d%H)
model_name=PatchTST
model_id=$DATE
exp_id="${DATE}_TEST"_individual

if [ ! -d "./logs/$exp_id" ]; then
    mkdir -p ./logs/$exp_id
fi

seq_len=256
pred_len=16
label_len=0

root_path_name="/ailab_mat/dataset/PV/DKASC/processed_data_day/"
data_name=DKASC
random_seed=2024


e_layers=8
n_heads=8
d_model=128
d_ff=512
patch_len=16

export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2 # 총 프로세스 수
export MASTER_ADDR='localhost'
export MASTER_PORT='12356'  # 임의의 빈 포트
export SCRIPT_NAME=$(basename "$0" .sh)

torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT run_longExp.py \
    --checkpoints "$TEST_{SCRIPT_NAME}" \
    --devices "0,1" \
    --individual 1 \
    --random_seed $random_seed \
    --is_pretraining 1 \
    --root_path $root_path_name \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len\
    --label_len $label_len \
    --pred_len 8\
    --enc_in 5 \
    --e_layers $e_layers \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.05\
    --fc_dropout 0.05\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --embed 'timeF' \
    --distributed \
    --wandb
    --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/$exp_id/$model_name'_'$data_name'_'$seq_len'_'$pred_len'_'$n_heads'_'$patch_len.log