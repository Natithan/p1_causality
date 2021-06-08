source $HOME/.bashrc
cd DeVLBert || exit
if [ "$HOSTNAME" = gimli ];
then
  conda activate devlbert_torch181bash
else
  conda activate devlbert
fi

CKPT_PATH=/cw/working-gimli/nathan/devlbert_checkpunten/epoch=2-step=30251-0.3.ckpt
SAVE_NAME=debug_ir

python train_tasks.py --bert_model bert-base-uncased --from_pretrained $CKPT_PATH  --config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 0 --tasks 3 --save_name $SAVE_NAME --use_ema --ema_decay_ratio 0.9999 --batch_size 8
