export CUDA_VISIBLE_DEVICES=1

root_path=/home/user/daojun/Data/TS/exchange_rate
model_name=Minusformer
dataset=Exchange
seq_len=96

python -u run.py \
  --is_training 1 \
  --root_path  $root_path \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128\
  --d_ff 128\
  --attn 0 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_S_'96.log


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --batch_size  16 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_S_'192.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name  \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 6 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.00005 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_S_'336.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name  \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --attn 0 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_S_'720.log