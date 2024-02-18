export CUDA_VISIBLE_DEVICES=1

root_path=/home/user/daojun/Data/TS/ETT-small/
model_name=Minusformer
dataset=ETTh2
seq_len=96

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 2 \
  --des 'Exp' \
  --d_model 128\
  --d_ff 128\
  --attn 0 \
  --learning_rate 0.0005 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'96.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --des 'Exp' \
  --attn 0 \
  --learning_rate 0.0005 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'192.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --des 'Exp' \
  --attn 0 \
  --batch_size 16 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'336.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --des 'Exp' \
  --attn 0 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'720.log  