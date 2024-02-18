export CUDA_VISIBLE_DEVICES=0

root_path=/home/user/daojun/Data/TS/traffic
model_name=Minusformer
seq_len=336

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path traffic.csv \
  --model_id traffic_336_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 6 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 8 \
  --itr 1 >logs/$model_name'_'traffic_$seq_len'_'96.log  


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path traffic.csv \
  --model_id traffic_336_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 16\
  --learning_rate 0.001\
  --itr 1  >logs/$model_name'_'traffic_$seq_len'_'192.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path traffic.csv \
  --model_id traffic_336_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 16\
  --learning_rate 0.001\
  --itr 1 >logs/$model_name'_'traffic_$seq_len'_'336.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path traffic.csv \
  --model_id traffic_336_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 6 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.001\
  --itr 1 >logs/$model_name'_'traffic_$seq_len'_'720.log 