python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 get_dic.py \
--from_pretrained bert-base-uncased \
--bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
--learning_rate 1e-4 --train_batch_size 8[] \
--save_name causal_pretrained --distributed --gpus 0 1 2 3


