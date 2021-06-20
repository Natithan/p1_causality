#!/bin/bash

#PBS -A lcalculus
#PBS -l nodes=1:ppn=9:gpus=1:skylake
#PBS -l partition=gpu
#PBS -l walltime=00:10:00
#PBS -l pmem=5gb
#PBS -m abe
#PBS -j oe
#PBS -M nathan.cornille@kuleuven.be
#PBS -N mini_finetune_vqa
#PBS -l qos=debugging

source $HOME/.bashrc
conda activate devlbert

CKPT_ROOT_DIR=/scratch/leuven/336/vsc33642
FINETUNED_DIR=debug_ckpts_downstream
PRETRAINED_CKPT_NAME=v6_devlbert_checkpunten/epoch=3-step=22687-0.3.ckpt
FINETUNED_CKPT_NAME=devlbert_vqa
FINETUNED_CKPT_EPOCH=1

# Finetuning on VQA
#
#python train_tasks.py --bert_model bert-base-uncased --from_pretrained $CKPT_ROOT_DIR/$PRETRAINED_CKPT_NAME --config_file config/bert_base_6layer_6conect.json --learning_rate 4e-5 --num_workers 0 --tasks 0 --save_name $FINETUNED_CKPT_NAME --use_ema --ema_decay_ratio 0.9999 --world_size 1 --batch_size 16 --output_dir $CKPT_ROOT_DIR/$FINETUNED_DIR --mini

# Evaluating on VQA
python eval_tasks.py --bert_model bert-base-uncased --from_pretrained  $CKPT_ROOT_DIR/$FINETUNED_DIR/VQA_bert_base_6layer_6conect-$FINETUNED_CKPT_NAME/pytorch_model_${FINETUNED_CKPT_EPOCH}_ema.bin  --config_file config/bert_base_6layer_6conect.json --tasks 0 --split test --save_name debug_devlbert_vqa --mini

