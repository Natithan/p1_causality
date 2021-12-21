#!/bin/bash
CKPT_ROOT_DIR=/cw/working-arwen/nathan
CODE_ROOT_DIR=/cw/liir/NoCsBack/testliir/nathan/p1_causality
PRETRAINED_CKPT_ROOT_DIR=$CKPT_ROOT_DIR/used_pretrain_ckpts
FINETUNED_CKPT_ROOT_DIR=$CKPT_ROOT_DIR/PP_bcorrect_ckpts
OUTPUT_ROOT_DIR=$CODE_ROOT_DIR/PP_bcorrect_output
cd $CODE_ROOT_DIR/DeVLBert
MINI=false
VILBERT=false
LIMIT_VISIBLE_GPUS=false
MAYBE_VISIBLE_GPUS=
USE_EXPLICIT_FT_EPOCH=false
EXPLICIT_FT_EPOCH=
maybe_debug=
USE_EXPLICIT_SAVE_NAME=false
EXPLICIT_SAVE_NAME=
while [[ $# -gt 0 ]]
do
  key="$1"
  echo "key"
  echo $key

  case $key in
      --only_eval)
      ONLY_EVAL=true
      shift # past argument
      ;;
      --mini)
      MINI=true
      shift # past argument
      ;;
      --visible_gpus)
      LIMIT_VISIBLE_GPUS=true
      MAYBE_VISIBLE_GPUS="--visible_gpus $2"
      shift # past argument
      shift # past value
      ;;
      --explicit_ft_epoch)
      USE_EXPLICIT_FT_EPOCH=true
      EXPLICIT_FT_EPOCH=$2
      shift # past argument
      shift # past value
      ;;
      --explicit_save_name)
      USE_EXPLICIT_SAVE_NAME=true
      EXPLICIT_SAVE_NAME=$2
      shift # past argument
      shift # past value
      ;;
      --debug)
      maybe_debug=echo
      shift # past argument
      ;;
      *) echo "Unknown parameter passed: $key"; exit 1 ;;
  esac
done





if [ "$MINI" = true ]
then
  echo 'mini was used'
  FINETUNED_CKPT_ROOT_DIR=${FINETUNED_CKPT_ROOT_DIR}_mini
  OUTPUT_ROOT_DIR=${OUTPUT_ROOT_DIR}_mini
  mini_arg='--mini'
  mini_prefix='mini_'
  max_t=60
  FINETUNED_CKPT_EPOCH=1
else
  echo 'mini was NOT used'
  mini_arg=''
  mini_prefix=''
  if [ $USE_EXPLICIT_FT_EPOCH = true ]; then
    FINETUNED_CKPT_EPOCH=$EXPLICIT_FT_EPOCH
  else
    FINETUNED_CKPT_EPOCH=19
  fi
  max_t=-1
fi
#PRETRAINED_CKPT_RUN_NAME=gimli_2
for PRETRAINED_CKPT_RUN_NAME in v6 # v4 literal_copy  # gimli_2 v6 gimli_1 v4
do
  case $PRETRAINED_CKPT_RUN_NAME in

    gimli_1)
      CKPT_FILE="pytorch_model_11.bin"
      ;;

    gimli_2)
      CKPT_FILE="epoch=11-step=110924-0.3.ckpt"
      ;;

    v4 | v5 | v6 | vilbert)
      CKPT_FILE="epoch=11-step=68063-0.3.ckpt"
      ;;

    literal_copy)
      CKPT_FILE="pytorch_model_11.bin"
      ;;

    vilbert_copy)
      CKPT_FILE="bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin"
      ;;

    *)
       echo "Unknown ckpt passed: $PRETRAINED_CKPT_RUN_NAME"; return 1 ;;
  esac
  PRETRAINED_CKPT=$PRETRAINED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME/$CKPT_FILE
  output_dir=$OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME
  finetuned_ckpt_dir=$FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME
  CFG_NAME=bert_base_6layer_6conect

  if [[ "$PRETRAINED_CKPT_RUN_NAME" == *"vilbert"* ]]; then
    echo $PRETRAINED_CKPT_RUN_NAME
    echo "Setting vilbert to true"
    VILBERT=true
  fi

  if [ $USE_EXPLICIT_SAVE_NAME = true ]
  then
      echo 'explicit save name was used'
      echo $EXPLICIT_SAVE_NAME
      SAVE_NAME=$EXPLICIT_SAVE_NAME
  else
      SAVE_NAME=best_val
  fi

    if [ "$VILBERT" = true ]
  then
      echo 'vilbert was used'
      echo $PRETRAINED_CKPT_RUN_NAME
      vilbert_arg='--vilbert'
  else
      vilbert_arg=''
  fi

#  $maybe_debug python eval_retrieval.py --bert_model bert-base-uncased --from_pretrained $PRETRAINED_CKPT --config_file config/bert_base_6layer_6conect.json --tasks 3 --split test --batch_size 1 --zero_shot $mini_arg --output_dir $output_dir/ZSIR --save_name default $vilbert_arg
#  echo "python $CODE_ROOT_DIR/test_confounder_finding.py --checkpoint $PRETRAINED_CKPT --out_dir $output_dir/mAP_output --max_t $max_t"
#  echo "python $CODE_ROOT_DIR/DeVLBert/eval_retrieval.py --bert_model bert-base-uncased --from_pretrained $PRETRAINED_CKPT --config_file config/bert_base_6layer_6conect.json --tasks 3 --split test --batch_size 1 --zero_shot $mini_arg --output_dir $output_dir/ZSIR --save_name default"

#  if ! [ "$ONLY_EVAL" = true ]
#  then
#    # Finetuning on VQA
#    b=256
#    g=1
#    a=16
#    $maybe_debug python $CODE_ROOT_DIR/DeVLBert/train_tasks.py --bert_model bert-base-uncased --from_pretrained $PRETRAINED_CKPT --config_file config/${CFG_NAME}.json --learning_rate 4e-5 --num_workers 0 --tasks 0 --save_name $PRETRAINED_CKPT_RUN_NAME --use_ema --ema_decay_ratio 0.9999 --world_size $g --batch_size $b --output_dir $finetuned_ckpt_dir $mini_arg --gradient_accumulation_steps $a $MAYBE_VISIBLE_GPUS
#  fi
#
#
#  # Evaluating on VQA
#  $maybe_debug python $CODE_ROOT_DIR/DeVLBert/eval_tasks.py --bert_model bert-base-uncased --from_pretrained  $finetuned_ckpt_dir/VQA_${CFG_NAME}-$PRETRAINED_CKPT_RUN_NAME/pytorch_model_${FINETUNED_CKPT_EPOCH}_ema.bin  --config_file config/bert_base_6layer_6conect.json --tasks 0 --split test --save_name $SAVE_NAME  $mini_arg --output_dir $output_dir/VQA $MAYBE_VISIBLE_GPUS --batch_size 100 $vilbert_arg

#  # Finetuning on IR
#  b=64
#  g=1
#  a=32
#
#  if ! [ "$ONLY_EVAL" = true ]
#  then
#    $maybe_debug python $CODE_ROOT_DIR/DeVLBert/train_tasks.py --bert_model bert-base-uncased --from_pretrained $PRETRAINED_CKPT --config_file config/${CFG_NAME}.json --learning_rate 4e-5 --tasks 3 --save_name $PRETRAINED_CKPT_RUN_NAME --use_ema --ema_decay_ratio 0.9999 --num_workers 0 --batch_size $b --output_dir $finetuned_ckpt_dir $mini_arg --world_size "$g" --gradient_accumulation_steps $a $MAYBE_VISIBLE_GPUS
#  fi
#
#
#  timestamp=$(date +%s.%N)
#  tmp_file=${finetuned_ckpt_dir}/tmp_${timestamp}
#  python3 $CODE_ROOT_DIR/get_best_val_run.py --run_name $PRETRAINED_CKPT_RUN_NAME --metric ir --tmp_file $tmp_file $mini_arg
#  FINETUNED_CKPT_PATH=`head $tmp_file`
#  rm $tmp_file
#  SAVE_NAME=best_val
#
#  echo "Best finetuned checkpoint path: "
#  echo $FINETUNED_CKPT_PATH
FINETUNED_CKPT_PATH=$finetuned_ckpt_dir/RetrievalFlickr30k_${CFG_NAME}-$PRETRAINED_CKPT_RUN_NAME/pytorch_model_${FINETUNED_CKPT_EPOCH}_ema.bin


  # Evaluating on IR
  $maybe_debug python $CODE_ROOT_DIR/DeVLBert/eval_retrieval.py --bert_model bert-base-uncased --from_pretrained "$FINETUNED_CKPT_PATH" --config_file config/${CFG_NAME}.json --tasks 3 --split test --batch_size 1 $mini_arg --output_dir $output_dir/IR  --save_name $SAVE_NAME $vilbert_arg  $MAYBE_VISIBLE_GPUS

#  qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"finetune_vqa $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir
#  if [ "$MINI" = true ]
#  then
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"mAP $PBS_SCRIPT_DIR/mAP.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --out_dir $MAP_OUT_DIR_NAME"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"zsir $PBS_SCRIPT_DIR/zsir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
##    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"finetune_vqa $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
##    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"finetune_ir $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#  else
##    qsub -l nodes=1:ppn=9:gpus=1:skylake,walltime=20:00:00 -N "${mini_prefix}"mAP $PBS_SCRIPT_DIR/mAP.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --out_dir $MAP_OUT_DIR_NAME"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,walltime=20:00:00 -N "${mini_prefix}"zsir $PBS_SCRIPT_DIR/zsir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#  fi
done