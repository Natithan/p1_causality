#!/bin/bash

#PBS -A lcalculus
#PBS -l nodes=1:ppn=16:gpus=4:cascadelake
#PBS -l partition=gpu
#PBS -l walltime=168:00:00
#PBS -l pmem=5gb
#PBS -m abe
#PBS -j oe
#PBS -M nathan.cornille@kuleuven.be
#PBS -N cascadelake_full_v4

source $HOME/.bashrc
conda activate devlbert


## First 12 epochs of training, with region mask probability 0.15
#python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .15 --trainer.auto_select_gpus false --trainer.gpus 4 --train_batch_size 128 --checkpoint_every_n_train_steps 0 --output_dir /scratch/leuven/336/vsc33642/v4_devlbert_checkpunten

## Further 12 epochs of training, with region mask probability 0.3
python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .3 --trainer.auto_select_gpus false --pt2_run --trainer.gpus 4 --train_batch_size 128 --checkpoint_every_n_train_steps 0 --output_dir /scratch/leuven/336/vsc33642/v4_devlbert_checkpunten

## Further 12 epochs of training, with region mask probability 0.3
#python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .15
#
## Finetuning on IR
#python train_tasks.py --bert_model bert-base-uncased \
#--from_pretrained save/devlbert/pytorch_model_11.bin \
#--config_file config/bert_base_6layer_6conect.json --learning_rate 4e-5 \
#--tasks 3 --save_name devlbert_i --use_ema --ema_decay_ratio 0.9999 --num_workers 1 --batch_size 16
#
## Evaluating on IR
#python eval_retrieval.py --bert_model bert-base-uncased \
#--from_pretrained /cw/working-gimli/nathan/ckpts_downstream/RetrievalFlickr30k_bert_base_6layer_6conect-24_ep_devlbert_i/pytorch_model_11_ema.bin \
#--config_file config/bert_base_6layer_6conect.json --tasks 3 --split test --batch_size 1
#
