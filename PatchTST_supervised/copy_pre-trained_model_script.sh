##
src_exp_id='240122_3s_04'
tgt_exp_id='240125_3stoSO_01'

if [ ! -d 'checkpoints/'$tgt_exp_id ]; then
    mkdir 'checkpoints/'$tgt_exp_id
fi

## 
src_model='PatchTST'
src_data='pv_DKASC_multi'
src_features='M'
src_d_model=128
src_n_heads=16
src_e_layers=3
src_d_layers=1
src_d_ff=256
src_factor=1
src_embed='timeF'
src_distil='True'
src_des='Exp_0'

tgt_model='PatchTST'
tgt_data='pv_SolarDB'       ##  pv_SolarDB  ,  pv_GIST
tgt_features='M'
tgt_d_model=128
tgt_n_heads=16
tgt_e_layers=3
tgt_d_layers=1
tgt_d_ff=256
tgt_factor=1
tgt_embed='timeF'
tgt_distil='True'
tgt_des='Exp_0'


##
for seq_len in 24 336
do
for label_len in 0
do
for pred_len in 24 48 96 192
do

src_model_id=$src_exp_id'_'$seq_len'_'$pred_len
tgt_model_id=$tgt_exp_id'_'$seq_len'_'$pred_len

src_path='checkpoints/'$src_exp_id'/'$src_model_id'_'$src_model'_'$src_data'_ft'$src_features'_sl'$seq_len'_ll'$label_len'_pl'$pred_len'_dm'$src_d_model'_nh'$src_n_heads'_el'$src_e_layers'_dl'$src_d_layers'_df'$src_d_ff'_fc'$src_factor'_eb'$src_embed'_dt'$src_distil'_'$src_des
tgt_path='checkpoints/'$tgt_exp_id'/'$tgt_model_id'_'$tgt_model'_'$tgt_data'_ft'$tgt_features'_sl'$seq_len'_ll'$label_len'_pl'$pred_len'_dm'$tgt_d_model'_nh'$tgt_n_heads'_el'$tgt_e_layers'_dl'$tgt_d_layers'_df'$tgt_d_ff'_fc'$tgt_factor'_eb'$tgt_embed'_dt'$tgt_distil'_'$tgt_des

if [ ! -d $tgt_path ]; then
    mkdir $tgt_path
fi

cp $src_path'/model_latest.pth' $tgt_path'/model_latest.pth'

done
done
done