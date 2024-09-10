# STNET 

## Getting Started

### Self-supervised Learning

1. Follow the first 2 steps above

2. Pre-training: The scirpt stnet_pretrain.py is to train the STNet. To run the code with a single GPU on ettm1, just run the following command
```
python stnet_pretrain.py --dset ettm1 --mask_ratio 0.4
```
The model will be saved to the saved_model folder for the downstream tasks. There are several other parameters can be set in the stnet_pretrain.py script.

 3. Fine-tuning: The script stnet_finetune.py is for fine-tuning step. Either linear_probing or fine-tune the entire network can be applied.
```
python stnet_finetune.py --dset ettm1 --pretrained_model <model_name>
```
