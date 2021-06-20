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
for PRETRAINED_CKPT_RUN_NAME in no_prior # gimli_2 v6 gimli_1 v4 vilbert literal_copy no_prior dependent_prior
do
  case $PRETRAINED_CKPT_RUN_NAME in

    gimli_1)
      CKPT_FILE="pytorch_model_11.bin"
      ;;

    gimli_2)
      CKPT_FILE="epoch=11-step=110924-0.3.ckpt"
      ;;

    v4 | v5 | v6 | vilbert | no_prior | dependent_prior)
      CKPT_FILE="epoch=11-step=68063-0.3.ckpt"
      ;;

    literal_copy)
      CKPT_FILE="pytorch_model_11.bin"
      ;;

    *)
       echo "Unknown ckpt passed: $PRETRAINED_CKPT_RUN_NAME"; return 1 ;;
  esac
  PRETRAINED_CKPT=$PRETRAINED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME/$CKPT_FILE
  if [ "$PRETRAINED_CKPT_RUN_NAME" = vilbert ]
  then
    echo "Setting vilbert to true"
    maybe_vilbert='--vilbert'
  else
    maybe_vilbert=''
  fi

  finetune_world_size=4
  eval_world_size=1
  finetune_ppn=$(expr $finetune_world_size \* 9)
  eval_ppn=$(expr $eval_world_size \* 9)
  if [ "$MINI" = true ]
  then
    finetune_world_size=1
    finetune_ppn=$(expr $finetune_world_size \* 9)
    QSUB_L_ARGS_MAP="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,qos=debugging,walltime=00:10:00"
    QSUB_L_ARGS_AA="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,qos=debugging,walltime=00:10:00"
    QSUB_L_ARGS_ZSIR="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,qos=debugging,walltime=00:10:00"
    QSUB_L_ARGS_IR="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,qos=debugging,walltime=00:10:00"
    QSUB_L_ARGS_VQA="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,qos=debugging,walltime=00:10:00"
    QSUB_L_ARGS_IR_EVAL="nodes=1:ppn=$eval_ppn:gpus=$eval_world_size:skylake,qos=debugging,walltime=00:10:00"
    QSUB_L_ARGS_VQA_EVAL="nodes=1:ppn=$eval_ppn:gpus=$eval_world_size:skylake,qos=debugging,walltime=00:10:00"
  else
    QSUB_L_ARGS_MAP="nodes=1:ppn=9:gpus=1:skylake,walltime=24:00:00"
    QSUB_L_ARGS_AA="nodes=1:ppn=9:gpus=1:skylake,walltime=12:00:00"
    QSUB_L_ARGS_ZSIR="nodes=1:ppn=9:gpus=1:skylake,walltime=06:00:00"
    QSUB_L_ARGS_IR="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,walltime=72:00:00"
    QSUB_L_ARGS_VQA="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,walltime=55:00:00"
    QSUB_L_ARGS_IR_EVAL="nodes=1:ppn=$eval_ppn:gpus=$eval_world_size:skylake,walltime=06:00:00"
    QSUB_L_ARGS_VQA_EVAL="nodes=1:ppn=$eval_ppn:gpus=$eval_world_size:skylake,walltime=06:00:00"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"mAP $PBS_SCRIPT_DIR/mAP.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --out_dir $MAP_OUT_DIR_NAME"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"zsir $PBS_SCRIPT_DIR/zsir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg $maybe_vilbert"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"finetune_vqa $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg $maybe_vilbert"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"finetune_ir $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg $maybe_vilbert"
  fi
#  qsub -l ${QSUB_L_ARGS_MAP} -N "${mini_prefix}"mAP_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/mAP.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#  qsub -l ${QSUB_L_ARGS_AA} -N "${mini_prefix}"avgAtt_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/avgAtt.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#  qsub -l ${QSUB_L_ARGS_ZSIR} -N "${mini_prefix}"zsir_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/zsir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg $maybe_vilbert"
#  qsub -l ${QSUB_L_ARGS_VQA} -N "${mini_prefix}"vqa_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --world_size $finetune_world_size $maybe_vilbert"
  qsub -l ${QSUB_L_ARGS_IR} -N "${mini_prefix}"ir_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --world_size $finetune_world_size $maybe_vilbert"
#  qsub -l ${QSUB_L_ARGS_VQA_EVAL} -N "${mini_prefix}"vqa_eval_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --world_size $eval_world_size --only_eval $mini_arg $maybe_vilbert"
#  qsub -l ${QSUB_L_ARGS_IR_EVAL} -N "${mini_prefix}"ir_eval_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --world_size $eval_world_size $maybe_vilbert --only_eval"

done