export CUDA_VISIBLE_DEVICES=0

model_name=MinusAutoformer
root_path=/home/user/daojun/Data/TS/electricity
seq_len=96


python -u run.py \
  --is_training 1 \
  --root_path /home/user/daojun/Data/TS/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16\
  --learning_rate 0.0005\
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_'96.log  

python -u run.py \
  --is_training 1 \
  --root_path /home/user/daojun/Data/TS/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16\
  --learning_rate 0.0005\
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_'192.log  


python -u run.py \
  --is_training 1 \
  --root_path /home/user/daojun/Data/TS/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16\
  --learning_rate 0.0005\
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_'336.log 


python -u run.py \
  --is_training 1 \
  --root_path /home/user/daojun/Data/TS/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16\
  --learning_rate 0.0005\
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_'720.log 