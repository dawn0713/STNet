if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=512
model_name=STNet

root_path_name=../datasets/
data_path_name=electricity.csv
model_id_name=Electricity

for pred_len in 96 192 336 720
do
    python -u stnet_finetune.py \
      --is_linear_probe 0 \
      --is_finetune 1 \
      --dset_finetune 'electricity' \
      --target_points $pred_len \
      --batch_size 32 \
      --n_epochs_finetune 20 \
      --pretrained_model './saved_models/electricity/masked_stnet/based_model/stnet_pretrained_cw512_patch12_stride12_epochs-pretrain10_mask0.4_model1.pth' \
      --finetuned_model_id $pred_len >logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done