CUDA_VISIBLE_DEVICES=0 python eval_retrieval.py \
--bert_model bert-base-uncased \
--from_pretrained save/devlbert/ir-pytorch_model_11_ema.bin \
--config_file config/bert_base_6layer_6conect.json \
--tasks 3 --split test --batch_size 1