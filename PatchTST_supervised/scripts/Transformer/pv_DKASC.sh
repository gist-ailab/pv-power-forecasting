# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# exp_id=debug
exp_id='231201_T01'

if [ ! -d "./logs/$exp_id" ]; then
    mkdir ./logs/$exp_id
fi

random_seed=2021
model_name=Transformer

for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --gpu 0 \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path ./dataset/pv/ \
    --data_path '91-Site_DKA-M9_B-Phase.csv' \
    --model_id pv_DKASC_$exp_id'_96_'$pred_len \
    --model $model_name \
    --data pv_DKASC \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 5 \
    --des 'Exp' \
    --itr 1 \
    --exp_id $exp_id \
    --train_epochs 1 >logs/$exp_id/DKASC_$model_name'_96_'$pred_len.log
done

# for model_name in Autoformer Informer Transformer
# do 
# for pred_len in 24 36 48 60
# do
#   python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path ./dataset/ \
#     --data_path national_illness.csv \
#     --model_id ili_36_$pred_len \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 36 \
#     --label_len 18 \
#     --pred_len $pred_len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1 >logs/LongForecasting/$model_name'_ili_'$pred_len.log
# done
# done
