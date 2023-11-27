# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

exp_id=exp2

if [ ! -d "./logs/$exp_id" ]; then
    mkdir ./logs/$exp_id
fi

seq_len=336
model_name=DLinear


python -u run_longExp.py \
  --gpu 0 \
  --is_training 1 \
  --root_path ./dataset/pv/ \
  --data_path 91-Site_DKA-M9_B-Phase.csv \
  --model_id pv_DKASC_$exp_id_$seq_len'_'96 \
  --model $model_name \
  --data pv \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --embed 'fixed' \
  --exp_id $exp_id \
  --itr 1 --batch_size 16  >logs/$exp_id/DKASC_$exp_id_$model_name'_'pv_$seq_len'_'96.log

python -u run_longExp.py \
  --gpu 0 \
  --is_training 1 \
  --root_path ./dataset/pv/ \
  --data_path 91-Site_DKA-M9_B-Phase.csv \
  --model_id pv_DKASC_$exp_id_$seq_len'_'192 \
  --model $model_name \
  --data pv \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --des 'Exp' \
  --embed 'fixed' \
  --exp_id $exp_id \
  --itr 1 --batch_size 16  >logs/$exp_id/DKASC_$exp_id_$model_name'_'pv_$seq_len'_'192.log

python -u run_longExp.py \
  --gpu 0 \
  --is_training 1 \
  --root_path ./dataset/pv/ \
  --data_path 91-Site_DKA-M9_B-Phase.csv \
  --model_id pv_DKASC_$exp_id_$seq_len'_'336 \
  --model $model_name \
  --data pv \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --des 'Exp' \
  --embed 'fixed' \
  --exp_id $exp_id \
  --itr 1 --batch_size 16  >logs/$exp_id/DKASC_$exp_id_$model_name'_'pv_$seq_len'_'336.log

python -u run_longExp.py \
  --gpu 0 \
  --is_training 1 \
  --root_path ./dataset/pv/ \
  --data_path 91-Site_DKA-M9_B-Phase.csv \
  --model_id pv_DKASC_$exp_id_$seq_len'_'720 \
  --model $model_name \
  --data pv \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --des 'Exp' \
  --embed 'fixed' \
  --exp_id $exp_id \
  --itr 1 --batch_size 16  >logs/$exp_id/DKASC_$exp_id_$model_name'_'pv_$seq_len'_'720.log

