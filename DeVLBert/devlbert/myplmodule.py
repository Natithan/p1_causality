import pytorch_lightning as pl
import os
import glob
import json

from .datasets import ConceptCapLoaderTrain

from util import is_master_rank, my_maybe_print, myprint, get_world_size
from datetime import datetime
from pytz import timezone

from constants import PROFILING_LOG_FILE_HANDLE, memprof_log_handle_for_name
# from memory_profiler import profile
from my_lmdb import get_core_ds
from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.optimization import BertAdam



class PLBertForMultimodalPretraining(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super(PLBertForMultimodalPretraining, self).__init__(*args,**kwargs)

    def setup(self, stage):
        if stage == 'fit':
            self.num_train_optimization_steps = (
                    int(
                        len(self.train_dataloader())
                        / self.args.gradient_accumulation_steps
                    )
                    * self.args.num_train_epochs
            )
        # self.save_hyperparameters()


    def on_save_checkpoint(self, checkpoint):
        checkpoint['args'] = self.args
        if self.args.checkpoint_every_n_train_steps > 0:
            checkpoint['batch_idx'] = self.trainer.batch_idx
            current_global_step = checkpoint['global_step'] - 1
            # Override the behaviour of starting at the next epoch when resuming from a mid-epoch checkpoint
            checkpoint['epoch'] = self.trainer.current_epoch
            core_ds = get_core_ds(self.train_dataset)
            steps_per_epoch = core_ds.nb_records // (self.args.train_batch_size * core_ds.nb_processes)
            myprint(checkpoint['global_step'], current_global_step - (steps_per_epoch * checkpoint['epoch']), checkpoint['batch_idx'])
            if not (current_global_step - (steps_per_epoch * checkpoint['epoch']) == checkpoint['batch_idx']):
                myprint(f"WARNING: current_global_step ({current_global_step}) - (steps_per_epoch ({steps_per_epoch}) * checkpoint['epoch'] ({checkpoint['epoch']})) != checkpoint['batch_idx'] ({checkpoint['batch_idx']})")
            myprint(f"Saving checkpoint with "
                    f"{checkpoint['epoch']} epochs, "
                    f"{checkpoint['global_step']} global step and "
                    f"{checkpoint['batch_idx']} batch_idx (per-epoch step)")
            if self.trainer.is_global_zero:
                myprint(self.trainer.is_global_zero)
                self.remove_old_ckpts()



    def remove_old_ckpts(self):

        myprint(f"Trying to remove checkpoints")
        list_of_ckpt = glob.glob(
            f'{self.args.output_dir}/epoch={self.trainer.current_epoch}*{self.args.region_mask_prob}.ckpt')  # * means all if need specific format then *.csv
        ckpts_to_remove = sorted(list_of_ckpt, key=os.path.getctime, reverse=True)[
                          2:]  # Keep only last 2 checkpoints for current epoch

        myprint(f"Removing old checkpoints: {ckpts_to_remove}")
        for ckpt in ckpts_to_remove:
            os.remove(ckpt)

    def on_load_checkpoint(self, checkpoint):

        if 'args' in checkpoint:
            ckpt_args = checkpoint['args']
            effective_batch_size = self.args.train_batch_size * self.args.trainer.gpus * self.args.trainer.accumulate_grad_batches
            ckpt_effective_batch_size = ckpt_args.train_batch_size * ckpt_args.trainer.gpus * ckpt_args.trainer.accumulate_grad_batches
            print("EFFECTIVE:",self.args.train_batch_size,self.args.trainer.gpus,self.args.trainer.accumulate_grad_batches)
            print("CKPT:",ckpt_args.train_batch_size,ckpt_args.trainer.gpus,ckpt_args.trainer.accumulate_grad_batches)
            assert effective_batch_size == ckpt_effective_batch_size
        if self.args.checkpoint_every_n_train_steps > 0:
            if 'batch_idx' in checkpoint and (self.trainer is not None): # second predicate for the case where we load not to recover, but to continue pt2 from pt1
                self.trainer.batch_idx = checkpoint['batch_idx']
                data_idx = self.batch_count_to_lmdb_index()
                myprint(f"Loading from checkpoint:  with "
                        f"{checkpoint['epoch']} epochs, "
                        f"{checkpoint['global_step']} global step and "
                        f"{checkpoint['batch_idx']} batch_idx (per-epoch step)\r\n"
                        f"batch_idx translated to data_idx {data_idx}")
                get_core_ds(self.train_dataset).data_index = data_idx

    def train_dataloader(self):
        if (not hasattr(self,'train_dataset')) or self.train_dataset is None:
            tokenizer = BertTokenizer.from_pretrained(
                self.args.bert_model, do_lower_case=self.args.do_lower_case
            )
            self.train_dataset = ConceptCapLoaderTrain(
                tokenizer,
                seq_len=self.args.max_seq_length,
                batch_size=self.args.train_batch_size,
                predict_feature=self.args.predict_feature,
                num_workers=self.args.num_workers,
                mini=self.args.mini,
                shuffle=self.args.shuffle,
                args=self.args
            )

            myprint("  Num examples =", self.train_dataset.num_dataset,
                    "  Batch size =", self.args.train_batch_size,
                    "  Num steps =", len(self.train_dataset))
        return self.train_dataset

    def batch_count_to_lmdb_index(self):
        train_batch_size, batch_idx = self.args.train_batch_size, self.trainer.batch_idx
        total_batch_size = train_batch_size * get_world_size()
        done_entries = batch_idx * total_batch_size
        txn_nonkey_sizes = get_core_ds(self.train_dataset).only_data_sizes
        old_sm = 0
        for txn_idx,s in enumerate(txn_nonkey_sizes):
            new_sm = old_sm + s
            if new_sm > done_entries:
                break
            old_sm = new_sm
        done_txn_yield_count = done_entries - old_sm
        return {'txn_idx': txn_idx,
                'txn_yield_count': done_txn_yield_count}

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        raise NotImplementedError
        # parser.add_argument('--encoder_layers', type=int, default=12) #TODO
        # parser.add_argument('--data_path', type=str, default='/some/path')
        return parent_parser


    def configure_optimizers(self):
        bert_weight_name = json.load(open("config/" + "bert-base-uncased_weight_name.json", "r"))

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        if self.args.freeze != -1:
            bert_weight_name_filtered = []
            for name in bert_weight_name:
                if 'embeddings' in name:
                    bert_weight_name_filtered.append(name)
                elif 'encoder' in name:
                    layer_num = name.split('.')[2]
                    if int(layer_num) <= self.args.freeze:
                        bert_weight_name_filtered.append(name)

            for key, value in dict(self.named_parameters()).items():
                if key[12:] in bert_weight_name_filtered:
                    value.requires_grad = False
        if not self.args.from_pretrained:
            param_optimizer = list(self.named_parameters())
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = []
            for key, value in dict(self.named_parameters()).items():
                if value.requires_grad:
                    # if key[12:] in bert_weight_name:
                    if key[5:] in bert_weight_name:  # Nathan: starts with "bert.", guess the 12: was for an old version
                        lr = self.args.learning_rate * 0.1
                    else:
                        lr = self.args.learning_rate

                    if any(nd in key for nd in no_decay):
                        optimizer_grouped_parameters += [
                            {"params": [value], "lr": lr, "weight_decay": 0.01}
                        ]

                    if not any(nd in key for nd in no_decay):
                        optimizer_grouped_parameters += [
                            {"params": [value], "lr": lr, "weight_decay": 0.0}
                        ]
            assert len(list(self.named_parameters())) == len(optimizer_grouped_parameters)

        num_train_optimization_steps = (
                int(
                    len(self.train_dataloader())
                    / self.args.gradient_accumulation_steps
                )
                * self.args.num_train_epochs
        )

        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            warmup=self.args.warmup_proportion,
            t_total=num_train_optimization_steps,
        )
        return optimizer
        # return torch.optim.Adam(self.parameters(),lr=self.args.learning_rate)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict['t'] = datetime.now(timezone('Europe/Brussels')).strftime("%X")
        return tqdm_dict


def get_core_module(module):
    from . import devlbert, vilbert
    core_module = module
    while not (isinstance(core_module, devlbert.BertForMultiModalPreTraining) or isinstance(core_module, vilbert.BertForMultiModalPreTraining) or isinstance(core_module, devlbert.DeVLBertForVLTasks)):
        core_module = core_module.module
    return core_module