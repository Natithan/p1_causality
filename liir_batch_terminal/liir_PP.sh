#!/bin/bash
CKPT_ROOT_DIR=/cw/working-gimli/nathan
CODE_ROOT_DIR=/cw/liir/NoCsBack/testliir/nathan/p1_causality
PRETRAINED_CKPT_ROOT_DIR=$CKPT_ROOT_DIR/used_pretrain_ckpts
FINETUNED_CKPT_ROOT_DIR=$CKPT_ROOT_DIR/PP_ckpts
OUTPUT_ROOT_DIR=$CODE_ROOT_DIR/PP_output
MINI=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mini) MINI=true; shift ;;
        *) echo "Unknown parameter passed: $1"; return 1 ;;
    esac
    shift
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
  FINETUNED_CKPT_EPOCH=19
  max_t=-1
fi
#PRETRAINED_CKPT_RUN_NAME=gimli_2
for PRETRAINED_CKPT_RUN_NAME in vilbert_copy # gimli_2 v6 gimli_1 v4
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
    VILBERT=true
  fi
  if [ "$VILBERT" = true ]
  then
      echo 'vilbert was used'
      vilbert_arg='--vilbert'
  else
      vilbert_arg=''
  fi


  echo "python eval_retrieval.py --bert_model bert-base-uncased --from_pretrained $PRETRAINED_CKPT --config_file config/bert_base_6layer_6conect.json --tasks 3 --split test --batch_size 1 --zero_shot $mini_arg --output_dir $output_dir/ZSIR --save_name default $vilbert_arg"
#  echo "python $CODE_ROOT_DIR/test_confounder_finding.py --checkpoint $PRETRAINED_CKPT --out_dir $output_dir/mAP_output --max_t $max_t"
#  echo "python $CODE_ROOT_DIR/DeVLBert/eval_retrieval.py --bert_model bert-base-uncased --from_pretrained $PRETRAINED_CKPT --config_file config/bert_base_6layer_6conect.json --tasks 3 --split test --batch_size 1 --zero_shot $mini_arg --output_dir $output_dir/ZSIR --save_name default"

#  # Finetuning on VQA
#  echo "python $CODE_ROOT_DIR/DeVLBert/train_tasks.py --bert_model bert-base-uncased --from_pretrained $PRETRAINED_CKPT --config_file config/${CFG_NAME}.json --learning_rate 4e-5 --num_workers 0 --tasks 0 --save_name $PRETRAINED_CKPT_RUN_NAME --use_ema --ema_decay_ratio 0.9999 --world_size 2 --batch_size 16 --output_dir $finetuned_ckpt_dir $mini_arg"
#
#  # Evaluating on VQA
#  echo "python $CODE_ROOT_DIR/DeVLBert/eval_tasks.py --bert_model bert-base-uncased --from_pretrained  $finetuned_ckpt_dir/VQA_${CFG_NAME}-$PRETRAINED_CKPT_RUN_NAME/pytorch_model_${FINETUNED_CKPT_EPOCH}_ema.bin  --config_file config/bert_base_6layer_6conect.json --tasks 0 --split test --save_name default  $mini_arg --output_dir $output_dir/VQA"

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