source $HOME/.bashrc
cd DeVLBert
conda activate devlbert_torch181
# First 12 epochs of training, with region mask probability 0.15
#python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .15 --trainer.gpus [1,2,3] --train_batch_size 96

## Further 12 epochs of training, with region mask probability 0.3
python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .3 --pt2_run --trainer.auto_select_gpus false --trainer.gpus -1 --train_batch_size 96 --checkpoint_every_n_train_steps 200 --mystepresume
#python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .3 --pt2_run --trainer.auto_select_gpus false --trainer.gpus -1 --train_batch_size 96 --checkpoint_every_n_train_steps 200 --mystepresume

