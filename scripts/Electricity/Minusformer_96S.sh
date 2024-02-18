export CUDA_VISIBLE_DEVICES=0

model_name=Minusformer
root_path=/home/user/daojun/Data/TS/electricity
seq_len=96


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.0005\
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_S_'96.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --learning_rate 0.00005\
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_S_'192.log  


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 6 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_S_'336.log 


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.001 \
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_S_'720.log 