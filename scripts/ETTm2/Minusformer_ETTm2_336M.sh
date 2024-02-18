export CUDA_VISIBLE_DEVICES=3

root_path=/home/user/daojun/Data/TS/ETT-small/
model_name=Minusformer
dataset=ETTm2
seq_len=336

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_96 \
  --model $model_name \
  --data $dataset \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 1 \
  --attn 0 \
  --batch_size 16 \
  --learning_rate 0.00005 \
  --itr 1 >logs/$model_name'_'ETTm2_$seq_len'_'96.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_192 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 1 \
  --attn 0 \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 8 \
  --learning_rate 0.00005 \
  --itr 1 >logs/$model_name'_'ETTm2_$seq_len'_'192.log


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_336 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 128 \
  --attn 0 \
  --learning_rate 0.00005 \
  --itr 1 >logs/$model_name'_'ETTm2_$seq_len'_'336.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_720 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 2 \
  --attn 0 \
  -batch_size 8 \
  --learning_rate 0.00005 \
  --itr 1 >logs/$model_name'_'ETTm2_$seq_len'_'720.log