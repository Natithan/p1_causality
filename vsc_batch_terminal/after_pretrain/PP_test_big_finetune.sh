#!/bin/bash
CKPT_ROOT_DIR=/scratch/leuven/336/vsc33642
CODE_ROOT_DIR=/data/leuven/336/vsc33642/p1_causality
PBS_SCRIPT_DIR=$CODE_ROOT_DIR/vsc_batch_terminal/after_pretrain
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
else
  echo 'mini was NOT used'
  mini_arg=''
  mini_prefix=''
fi
#PRETRAINED_CKPT_RUN_NAME=gimli_2
for PRETRAINED_CKPT_RUN_NAME in gimli_2 v6 gimli_1
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

    *)
       echo "Unknown ckpt passed: $PRETRAINED_CKPT_RUN_NAME"; return 1 ;;
  esac
  PRETRAINED_CKPT=$PRETRAINED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME/$CKPT_FILE

  qsub -l nodes=1:ppn=18:gpus=2:skylake,walltime=00:05:00 -N "${mini_prefix}"finetune_vqa $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
  qsub -l nodes=1:ppn=18:gpus=2:skylake,walltime=00:05:00 -N "${mini_prefix}"finetune_ir $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#  if [ "$MINI" = true ]
#  then
#    qsub -l nodes=1:ppn=9:gpus=2:skylake,walltime=00:10:00 -N "${mini_prefix}"finetune_vqa $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#    qsub -l nodes=1:ppn=9:gpus=2:skylake,walltime=00:10:00 -N "${mini_prefix}"finetune_ir $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#  else
#    qsub -l nodes=1:ppn=9:gpus=2:skylake,walltime=00:10:00 -N "${mini_prefix}"finetune_vqa $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#    qsub -l nodes=1:ppn=9:gpus=2:skylake,walltime=00:10:00 -N "${mini_prefix}"finetune_ir $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#  fi
done