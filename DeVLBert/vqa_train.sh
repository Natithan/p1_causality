python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py \
--bert_model bert-base-uncased --from_pretrained save/devlbert/pytorch_model_11.bin  \
--config_file config/bert_base_6layer_6conect.json --learning_rate 4e-5 --num_workers 16 \
--tasks 0 --save_name devlbert_vqa \
--use_ema --ema_decay_ratio 0.9999