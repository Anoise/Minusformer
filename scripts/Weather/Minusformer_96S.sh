export CUDA_VISIBLE_DEVICES=0

root_path=/home/user/daojun/Data/TS/weather/
model_name=Minusformer
seq_len=96


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --itr 1 >logs/$model_name'_'weather_$seq_len'_S_'96.log 

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --learning_rate 0.001 \
  --itr 1 >logs/$model_name'_'weather_$seq_len'_S_'192.log 


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --itr 1 >logs/$model_name'_'weather_$seq_len'_S_'336.log 


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 >logs/$model_name'_'weather_$seq_len'_S_'720.log 