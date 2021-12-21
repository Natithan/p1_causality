source $HOME/.bashrc
cd DeVLBert
if [[ $HOSTNAME = gimli ]]
then
  conda activate devlbert_torch181
else
  conda activate devlbert
fi


MODEL=dv
#MODEL=dv
#MODEL=no_prior
INDEX=8
batch_size=64
gpus=2
ga=4

LIMIT_VISIBLE_GPUS=false
MAYBE_VISIBLE_GPUS=
maybe_debug=
PT2_ONLY=false

while [[ $# -gt 0 ]]
do
  key="$1"
  echo "key"
  echo $key

  case $key in
      --visible_gpus)
      LIMIT_VISIBLE_GPUS=true
      MAYBE_VISIBLE_GPUS="--visible_gpus $2"
      shift # past argument
      shift # past value
      ;;
      --debug)
      maybe_debug=echo
      shift # past argument
      ;;
      --pt2_only)
      PT2_ONLY=true
      shift # past argument
      ;;
      *) echo "Unknown parameter passed: $key"; exit 1 ;;
  esac
done



name=${MODEL}${INDEX}_${HOSTNAME}_b${batch_size}g${gpus}a${ga}

declare -a COMMON_VARS=('--config' 'config/pretrain_concap_devlbert.yml' '--trainer.auto_select_gpus' 'false' '--trainer.gpus' ${gpus} '--train_batch_size' ${batch_size} '--trainer.accumulate_grad_batches' ${ga} '--checkpoint_every_n_train_steps' 0 '--output_dir' "/cw/working-arwen/nathan/${name}_checkpunten" "--run_name" "$name" $MAYBE_VISIBLE_GPUS)


case $MODEL in

  vi)
    COMMON_VARS+=("--vilbert")
    ;;

  no_prior)
    COMMON_VARS+=("--no_prior")
    ;;

  dep_prior)
    COMMON_VARS+=("--dependent_prior")
    ;;
esac

if [[ $MODEL = v ]]
then
  COMMON_VARS+=("--vilbert")
fi

if ! [ "$PT2_ONLY" = true ]; then
  # First 12 epochs of training, with region mask probability 0.15
  $maybe_debug python train_concap.py  ${COMMON_VARS[@]}  --region_mask_prob .15 --mystepresume
fi

## Further 12 epochs of training, with region mask probability 0.3
$maybe_debug python train_concap.py ${COMMON_VARS[@]} --region_mask_prob .3 --pt2_run

