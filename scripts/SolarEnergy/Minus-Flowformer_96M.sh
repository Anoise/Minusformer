export CUDA_VISIBLE_DEVICES=1

root_path=/home/user/daojun/Data/TS/Solar/
model_name=MinusFlowformer
dataset=Solar
seq_len=96


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --batch_size 8 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'96.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path solar_AL.txt \
  --model_id solar_96_192 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.0005\
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'192.log
 
python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path solar_AL.txt \
  --model_id solar_96_336 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --batch_size 8 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'336.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path solar_AL.txt \
  --model_id solar_96_720 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 1 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.001\
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'720.log
