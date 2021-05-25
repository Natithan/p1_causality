# Parameters not specified here are in DeVLBert/config/pretrain_concap_devlbert.yml
# first step
python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .15
# second step
# change region mask probability from 0.15 to 0.3
python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .3
