from my_lmdb import get_core_ds
from pytorch_lightning.callbacks import ProgressBar, ModelCheckpoint
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.supporters import prefetch_iterator
from util import myprint

from devlbert.myplmodule import get_core_module


class MyDataConnector(DataConnector):

    def __init__(self, trainer, exp_args):
        super().__init__(trainer)
        self.first_after_restart = True
        self.args = exp_args

    def get_profiled_train_dataloader(self, train_dataloader):
        # profiled_dl = self.trainer.profiler.profile_iterable(
        #     enumerate(prefetch_iterator(train_dataloader)), "get_train_batch"
        # )
        # return profiled_dl
        # Flow is:
        # 1) model.on_load_checkpoint: set trainer.batch_idx to the ckpt one
        # 2) go through this enumerate
        # 3) in training_loop.py, trainer.batch_idx is set equal to the number coming out of this enumerate
        # model = get_core_module(self.trainer.model)
        # core_ds = get_core_ds(model.train_dataset)
        # steps_per_epoch = core_ds.nb_records // (model.args.train_batch_size * core_ds.nb_processes)
        # if self.trainer.batch_idx == (steps_per_epoch - 1):
        #     myprint("resetting batch_idx for next epoch")
        #     self.trainer.batch_idx = -1

        if self.args.checkpoint_every_n_train_steps > 0:
            if self.first_after_restart:
                start = self.trainer.batch_idx + 1 if self.trainer.batch_idx > 0 else 0
                myprint(f"Starting enumerate at {start}")
                profiled_dl = self.trainer.profiler.profile_iterable(
                    enumerate(prefetch_iterator(train_dataloader), start=start), "get_train_batch"
                )
                self.first_after_restart = False
            else:
                myprint(f"Starting enumerate from scratch, assuming not in first epoch after restart.")
                profiled_dl = self.trainer.profiler.profile_iterable(
                    enumerate(prefetch_iterator(train_dataloader)), "get_train_batch"
                )
        else:
            profiled_dl = super().get_profiled_train_dataloader(train_dataloader)
        return profiled_dl


class MyProgressBar(ProgressBar):

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        # Update the progress bar with potentially mid-epoch batch index
        core_module = get_core_module(pl_module)
        if core_module.args.checkpoint_every_n_train_steps > 0:
            self.main_progress_bar.n = trainer.batch_idx
            self.main_progress_bar.refresh()

            self._train_batch_idx = trainer.batch_idx
            self.just_after_restarting_training = True

    def on_train_epoch_start(self, trainer, pl_module):
        core_module = get_core_module(pl_module)
        if core_module.args.checkpoint_every_n_train_steps > 0:
            if not self.just_after_restarting_training:
                self._train_batch_idx = 0
            else:
                self._train_batch_idx = trainer.batch_idx
                self.just_after_restarting_training = False
        else:
            self._train_batch_idx = 0



class MyModelCheckpoint(ModelCheckpoint):

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_epoch_end(trainer, pl_module, *args, **kwargs)
        # Save at the end of every epoch in any case
        self.save_checkpoint(trainer, pl_module)



    # def on_train_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, dataloader_idx,**kwargs):
    #     pl_module.batch_idx = batch_idx


