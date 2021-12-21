#!/bin/bash

#PBS -A lcalculus
#PBS -l partition=gpu
#PBS -l pmem=5gb
#PBS -m abe
#PBS -j oe
#PBS -M nathan.cornille@kuleuven.be

source $HOME/.bashrc
conda activate devlbert

MINI=false
VILBERT=false
LAST_CKPT=false
#LAST_CKPT=true
ONLY_EVAL=false
CFG_NAME=bert_base_6layer_6conect

# Adapted from https://stackoverflow.com/a/14203146/6297057
#POSITIONAL=()
while [[ $# -gt 0 ]]
do
  key="$1"
  echo "key"
  echo $key

  case $key in
      -p|--pretrained_ckpt)
      pretrained_ckpt="$2"
      shift # past argument
      shift # past value
      ;;
      -f|--finetuned_ckpt_dir)
      finetuned_ckpt_dir="$2"
      shift # past argument
      shift # past value
      ;;
      -o|--output_dir)
      output_dir="$2"
      shift # past argument
      shift # past value
      ;;
      -w|--world_size)
      world_size="$2"
      shift # past argument
      shift # past value
      ;;
      --vilbert)
      VILBERT=true
      shift # past argument
      ;;
      --only_eval)
      ONLY_EVAL=true
      shift # past argument
      ;;
      --mini)
      MINI=true
      shift # past argument
      ;;
  #    *)    # unknown option
  #    POSITIONAL+=("$1") # save it in an array for later
  #    shift # past argument
  #    ;;
      *) echo "Unknown parameter passed: $key"; exit 1 ;;
  esac
done
#set -- "${POSITIONAL[@]}" # restore positional parameters


PRETRAINED_CKPT_RUN_NAME=`basename "$finetuned_ckpt_dir"`

echo "pretrained_ckpt: ${pretrained_ckpt}"
if [ "$MINI" = true ]
then
    echo 'mini was used'
    mini_arg='--mini'
    FINETUNED_CKPT_EPOCH=1
    batch_size=8
    gradient_accumulation_steps=1
else
    mini_arg=''
    FINETUNED_CKPT_EPOCH=19
    batch_size=64 # This is the effective batch size
    gradient_accumulation_steps=2 # Otherwise won't fit on 4 16GB GPUs.
fi

if [ "$VILBERT" = true ]
then
    echo 'vilbert was used'
    vilbert_arg='--vilbert'
else
    vilbert_arg=''
fi


if ! [ "$ONLY_EVAL" = true ]
then
  # Finetuning on IR
  python train_tasks.py --bert_model bert-base-uncased --from_pretrained $pretrained_ckpt --config_file config/${CFG_NAME}.json --learning_rate 4e-5 --tasks 3 --save_name $PRETRAINED_CKPT_RUN_NAME --use_ema --ema_decay_ratio 0.9999 --num_workers 0 --batch_size $batch_size --output_dir $finetuned_ckpt_dir $mini_arg --world_size "$world_size" --gradient_accumulation_steps $gradient_accumulation_steps
fi

if [ "$LAST_CKPT" = true ]
then
    FINETUNED_CKPT_PATH=$finetuned_ckpt_dir/RetrievalFlickr30k_${CFG_NAME}-$PRETRAINED_CKPT_RUN_NAME/pytorch_model_${FINETUNED_CKPT_EPOCH}_ema.bin
    SAVE_NAME=default
else
    timestamp=$(date +%s.%N)
    tmp_file=${finetuned_ckpt_dir}/tmp_${timestamp}
    python3 ../get_best_val_run.py --run_name $PRETRAINED_CKPT_RUN_NAME --metric ir --tmp_file $tmp_file $mini_arg
    FINETUNED_CKPT_PATH=`head $tmp_file`
    rm $tmp_file
    SAVE_NAME=best_val
fi
echo "Best finetuned checkpoint path: "
echo $FINETUNED_CKPT_PATH


# Evaluating on IR
python eval_retrieval.py --bert_model bert-base-uncased --from_pretrained "$FINETUNED_CKPT_PATH" --config_file config/${CFG_NAME}.json --tasks 3 --split test --batch_size 1 $mini_arg --output_dir $output_dir/IR  --save_name $SAVE_NAME $vilbert_arg --visible_gpus 0

