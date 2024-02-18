export CUDA_VISIBLE_DEVICES=1

root_path=/home/user/daojun/Data/TS/PEMS/
model_name=Minusformer
dataset=PEMS03
seq_len=336

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path PEMS03.npz \
  --model_id PEMS03_336_12 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 12 \
  --e_layers 1 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'12.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path PEMS03.npz \
  --model_id PEMS03_336_24 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 24 \
  --e_layers 1 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'24.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path PEMS03.npz \
  --model_id PEMS03_336_36 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 36 \
  --e_layers 1 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'36.log


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path PEMS03.npz \
  --model_id PEMS03_336_48 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --e_layers 6 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --batch_size 8 \
  --learning_rate 0.00005 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'48.log