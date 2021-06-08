source $HOME/.bashrc

until python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .15  --mystepresume --checkpoint_every_n_train_steps 1; do
    echo "Part 1 training crashed with exit code $?.  Respawning in 10 seconds.." >&2
    sleep 10
done

## Further X epochs of training, with region mask probability 0.3


until python train_concap.py --config config/pretrain_concap_devlbert.yml --region_mask_prob .3 --pt2_run --mystepresume  --checkpoint_every_n_train_steps 100; do
    echo "Part 2 training crashed with exit code $?.  Respawning in 10 seconds.." >&2
    sleep 10
done