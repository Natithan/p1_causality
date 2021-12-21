#!/bin/bash
CKPT_ROOT_DIR=/scratch/leuven/336/vsc33642
CODE_ROOT_DIR=/data/leuven/336/vsc33642/p1_causality
PBS_SCRIPT_DIR=$CODE_ROOT_DIR/vsc_batch_terminal/after_pretrain
#PRETRAINED_CKPT_ROOT_DIR=$CKPT_ROOT_DIR/used_pretrain_b64_ckpts
#FINETUNED_CKPT_ROOT_DIR=$CKPT_ROOT_DIR/PP_b64_ckpts
#OUTPUT_ROOT_DIR=$CODE_ROOT_DIR/PP_b64_output
PRETRAINED_CKPT_ROOT_DIR=$CKPT_ROOT_DIR/used_pretrain_ckpts
FINETUNED_CKPT_ROOT_DIR=$CKPT_ROOT_DIR/PP_bcorrect_ckpts
OUTPUT_ROOT_DIR=$CODE_ROOT_DIR/PP_bcorrect_output
cd $PBS_SCRIPT_DIR
MINI=false
maybe_debug=
MAYBE_ONLY_EVAL=
#while [[ "$#" -gt 0 ]]; do
#    case $1 in
#        --mini) MINI=true; shift ;;
#        *) echo "Unknown parameter passed: $1"; return 1 ;;
#    esac
#    shift
#done

while [[ $# -gt 0 ]]
do
  key="$1"
  echo "key"
  echo $key

  case $key in
      --only_eval)
      MAYBE_ONLY_EVAL="--only_eval"
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
else
  echo 'mini was NOT used'
  mini_arg=''
  mini_prefix=''
fi
#PRETRAINED_CKPT_RUN_NAME=gimli_2
# gimli_2 v6 gimli_1 v4 v5 vilbert literal_copy vilbert_2 vilbert_3 vilbert_4 vilbert_5 no_prior dependent_prior
for PRETRAINED_CKPT_RUN_NAME in v6 dv7 dependent_prior # vi_2 no_prior dep_prior # dv_1 dv_2 dv_3 dv_4 dv_5 vi_1 vi_2 vi_3 vi_4 vi_5 no_prior dep_prior
do
  if [[  "$PRETRAINED_CKPT_ROOT_DIR" == *"b64"* ]]; then
      CKPT_FILE=$(ls -t $PRETRAINED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME | head -1)
      if [ -z "$CKPT_FILE" ]
      then
        echo "No checkpoints in : $PRETRAINED_CKPT_RUN_NAME"; return 1 ;
      fi
  else
    case $PRETRAINED_CKPT_RUN_NAME in

      gimli_1)
        CKPT_FILE="pytorch_model_11.bin"
        ;;

      gimli_2)
        CKPT_FILE="epoch=11-step=110924-0.3.ckpt"
        ;;

      dv7 | v4 | v6 | vilbert )
        CKPT_FILE="epoch=11-step=68063-0.3.ckpt"
        ;;

      no_prior | dependent_prior | vilbert_2 | vilbert_3 | vilbert_4 | vilbert_5)
        CKPT_FILE="epoch=11-step=68075-0.3.ckpt"
        ;;

      v5)
        CKPT_FILE="epoch=11-step=68071-0.3.ckpt"
        ;;

      literal_copy)
        CKPT_FILE="pytorch_model_11.bin"
        ;;

      *)
         echo "Unknown ckpt passed: $PRETRAINED_CKPT_RUN_NAME"; return 1 ;;
    esac
  fi
  PRETRAINED_CKPT=$PRETRAINED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME/$CKPT_FILE

  if [[  "$PRETRAINED_CKPT_RUN_NAME" == *"vi"* ]]
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
    QSUB_L_ARGS_IR="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,walltime=26:00:00"
    QSUB_L_ARGS_VQA="nodes=1:ppn=$finetune_ppn:gpus=$finetune_world_size:skylake,walltime=24:00:00"
    QSUB_L_ARGS_IR_EVAL="nodes=1:ppn=$eval_ppn:gpus=$eval_world_size:skylake,walltime=06:00:00"
    QSUB_L_ARGS_VQA_EVAL="nodes=1:ppn=$eval_ppn:gpus=$eval_world_size:skylake,walltime=06:00:00"
    if [[ $MAYBE_ONLY_EVAL = "--only_eval" ]]; then
      QSUB_L_ARGS_IR=$QSUB_L_ARGS_IR_EVAL
      QSUB_L_ARGS_VQA=$QSUB_L_ARGS_VQA_EVAL
      finetune_world_size=$eval_world_size
    fi
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"mAP $PBS_SCRIPT_DIR/mAP.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --out_dir $MAP_OUT_DIR_NAME"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"zsir $PBS_SCRIPT_DIR/zsir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg $maybe_vilbert"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"finetune_vqa $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg $maybe_vilbert"
#    qsub -l nodes=1:ppn=9:gpus=1:skylake,qos=debugging,walltime=00:10:00 -N "${mini_prefix}"finetune_ir $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg $maybe_vilbert"
  fi
#  if ! [ "$maybe_vilbert" = "--vilbert" ]; then
#    $maybe_debug qsub -l ${QSUB_L_ARGS_MAP} -N "${mini_prefix}"mAP_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/mAP.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#    $maybe_debug qsub -l ${QSUB_L_ARGS_AA} -N "${mini_prefix}"avgAtt_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/avgAtt.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg"
#  fi
#  $maybe_debug qsub -l ${QSUB_L_ARGS_ZSIR} -N "${mini_prefix}"zsir_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/zsir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg $maybe_vilbert"
#  $maybe_debug qsub -l ${QSUB_L_ARGS_VQA} -N "${mini_prefix}"vqa_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --world_size $finetune_world_size $maybe_vilbert $MAYBE_ONLY_EVAL"
  $maybe_debug qsub -l ${QSUB_L_ARGS_IR} -N "${mini_prefix}"ir_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --world_size $finetune_world_size $maybe_vilbert $MAYBE_ONLY_EVAL"
#  $maybe_debug qsub -l ${QSUB_L_ARGS_VQA_EVAL} -N "${mini_prefix}"vqa_eval_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/finetune_vqa.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --world_size $eval_world_size --only_eval $mini_arg $maybe_vilbert"
#  $maybe_debug qsub -l ${QSUB_L_ARGS_IR_EVAL} -N "${mini_prefix}"ir_eval_"${PRETRAINED_CKPT_RUN_NAME}" $PBS_SCRIPT_DIR/finetune_ir.pbs -F "--pretrained_ckpt $PRETRAINED_CKPT --finetuned_ckpt_dir $FINETUNED_CKPT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME --output_dir $OUTPUT_ROOT_DIR/$PRETRAINED_CKPT_RUN_NAME $mini_arg --world_size $eval_world_size $maybe_vilbert --only_eval"

done