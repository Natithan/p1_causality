from datetime import datetime

from io import open

import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_pretrained_bert import BertAdam
from time import time as t
import time
from torch import nn
import torch
from torch.utils.data import DataLoader

from devlbert.devlbert import BertPreTrainedModel
import numpy as np

from util import is_master_rank


class DummyModel(pl.LightningModule):
    """BERT model with multi modal pre-training heads.
    """

    def __init__(self):
        # super(BertForMultiModalPreTraining, self).__init__(config)
        # self.bert = BertModel(config)
        # # print(self.bert.embeddings.word_embeddings.weight.shape) #torch.Size([30522, 768])
        # self.cls = BertPreTrainingHeads(
        #     config, self.bert.embeddings.word_embeddings.weight
        # )
        #
        # self.apply(self.init_bert_weights)
        # self.predict_feature = config.predict_feature
        # self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        #
        # self.causal_v = Causal_v()
        # self.causal_t = Causal_t()
        # self.causal_t2v = Causal_t2v()
        # self.causal_v2t = Causal_v2t()
        # self.args = args
        # #
        # # print("model's option for predict_feature is ", config.predict_feature)
        #
        # if self.predict_feature:
        #     self.vis_criterion = nn.MSELoss(reduction="none")
        # else:
        #     self.vis_criterion = nn.KLDivLoss(reduction="none")
        # # self.save_hyperparameters()
        super().__init__()
        self.input_to_intermediate = nn.Linear(2048, 32768)
        # self.int_to_int = nn.Linear(32768, 32768)
        self.int_to_last = nn.Linear(32768, 2048)
        self.last_to_pred = nn.Linear(2048, 1)


    def forward(
            self,
            inputs, label
    ):
        pred = self.last_to_pred(self.int_to_last(self.input_to_intermediate(inputs)))
        loss = torch.nn.MSELoss(pred,label)
        return loss
        # # if str(input_ids.device) != 'cuda:0' or str(self.bert.embeddings.word_embeddings.weight.device) != 'cuda:0':
        # #     print(5)
        # # print("SLEEPING FOR DEBUGGING")
        # # sleep(10000000)
        # # in this model, we first embed the images.
        # sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
        #     input_ids,
        #     image_feat,
        #     image_loc,
        #     token_type_ids,
        #     attention_mask,
        #     image_attention_mask,
        #     output_all_encoded_layers=False,
        #     output_all_attention_masks=output_all_attention_masks
        # )
        # # print(sequence_output_t.shape, sequence_output_v.shape) # torch.Size([64, 36, 768]) torch.Size([64, 37, 1024])
        # # Added by jt
        #
        # if masked_lm_labels is not None and next_sentence_label is not None and image_target is not None:
        #
        #     causal_sequence_output_v = self.causal_v(sequence_output_v[:, 1:])
        #     causal_sequence_output_t = self.causal_t(sequence_output_t)
        #
        #     causal_sequence_output_v2t = self.causal_v2t(sequence_output_v[:, 1:])
        #     causal_sequence_output_t2v = self.causal_t2v(sequence_output_t)
        #
        #     prediction_scores_t, prediction_scores_v, seq_relationship_score, \
        #     causal_prediction_v_loss, causal_prediction_t_loss, \
        #     causal_prediction_v2t_loss, causal_prediction_t2v_loss = self.cls(
        #         sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v,
        #         causal_sequence_output_v, causal_sequence_output_v2t, causal_label_v, image_target,
        #         causal_sequence_output_t, causal_sequence_output_t2v, causal_label_t
        #     )
        #
        # else:
        #     prediction_scores_t, prediction_scores_v, seq_relationship_score = self.cls(
        #         sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        #     )
        #
        # if masked_lm_labels is not None and next_sentence_label is not None and image_target is not None:
        #     prediction_scores_v = prediction_scores_v[:, 1:]
        #
        #     if self.predict_feature:
        #         img_loss = self.vis_criterion(prediction_scores_v, image_target)
        #         masked_img_loss = torch.sum(
        #             img_loss * (image_label == 1).unsqueeze(2).float()
        #         ) / max(torch.sum((image_label == 1).unsqueeze(2).expand_as(img_loss)), 1)
        #     else:
        #         img_loss = self.vis_criterion(
        #             F.log_softmax(prediction_scores_v, dim=2), image_target
        #         )
        #         masked_img_loss = torch.sum(
        #             img_loss * (image_label == 1).unsqueeze(2).float()
        #         ) / max(torch.sum((image_label == 1)), 1)
        #
        #     masked_lm_loss = self.loss_fct(
        #         prediction_scores_t.view(-1, self.config.vocab_size),
        #         masked_lm_labels.view(-1),
        #     )
        #
        #     next_sentence_loss = self.loss_fct(
        #         seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
        #     )
        #
        #     return masked_lm_loss, masked_img_loss, next_sentence_loss, \
        #            causal_prediction_v_loss, causal_prediction_t_loss, \
        #            causal_prediction_v2t_loss, causal_prediction_t2v_loss
        # else:
        #     return prediction_scores_t, prediction_scores_v, seq_relationship_score, all_attention_mask

    def training_step(self, batch, batch_idx):
        # input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, \
        # image_ids, causal_label_t, causal_label_v = (
        #     batch
        # )
        # if batch_idx > 20:
        #     print("Stopping for debugging")
        # p = '/home/nathan_e_d_cornille_gmail_com/p1_causality/DeVLBert/tmp_check_ids.txt'
        # with open(p, 'a') as f:
        #     for image_id in image_ids:
        #         f.write(f'{image_id.cpu().numpy()}\n')

        # with open(p, 'r') as f:
        #     idss = f.readlines()
        #
        # len(set(idss))
        # s = t()
        #
        # # print_gpu_mem()
        # my_maybe_print(f'Doing Model input --> model output losses')
        # myprint(str(list(model.module.parameters())[0].device))
        # myprint(str(input_ids.device))
        # skip_sleep = False
        # if not default_gpu:
        #     print(5)
        # if not skip_sleep:
        #     print("SLEEPING FOR DEBUGGING")
        #     sleep(10000000)
        loss = self(*batch)
        # my_maybe_print(f'Model input --> model output losses took {t() - s}')
        #
        # masked_loss_v = masked_loss_v * self.args.img_weight
        # loss = masked_loss_t + masked_loss_v + next_sentence_loss + \
        #        causal_prediction_v_loss + causal_prediction_t_loss + \
        #        causal_prediction_v2t_loss + causal_prediction_t2v_loss

        # region tboard logging
        if is_master_rank() and self.trainer.logger_connector.should_update_logs:  # and not self.args.mini: # skip when debugging as this involves moving to cpu which takes quite a long time the first time round
            self.log("loss", loss)
        # endregion
        # else:
        #     myprint("*"*50, "STOPPING NON-MASTER PROCESSES FOR EASY DEBUGGING", "*"*50)
        #     stop = True
        #     while stop:
        #         sleep(10)

        return {'loss': loss}

    def configure_optimizers(self):
        return BertAdam(
            self.parameters(),
            lr=0.01)
    #     bert_weight_name = json.load(open("config/" + "bert-base-uncased_weight_name.json", "r"))
    #
    #     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #     if self.args.freeze != -1:
    #         bert_weight_name_filtered = []
    #         for name in bert_weight_name:
    #             if 'embeddings' in name:
    #                 bert_weight_name_filtered.append(name)
    #             elif 'encoder' in name:
    #                 layer_num = name.split('.')[2]
    #                 if int(layer_num) <= self.args.freeze:
    #                     bert_weight_name_filtered.append(name)
    #
    #         for key, value in dict(self.named_parameters()).items():
    #             if key[12:] in bert_weight_name_filtered:
    #                 value.requires_grad = False
    #     if not self.args.from_pretrained:
    #         param_optimizer = list(self.named_parameters())
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [
    #                     p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
    #                 ],
    #                 "weight_decay": 0.01,
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in param_optimizer if any(nd in n for nd in no_decay)
    #                 ],
    #                 "weight_decay": 0.0,
    #             },
    #         ]
    #     else:
    #         optimizer_grouped_parameters = []
    #         for key, value in dict(self.named_parameters()).items():
    #             if value.requires_grad:
    #                 # if key[12:] in bert_weight_name:
    #                 if key[5:] in bert_weight_name:  # Nathan: starts with "bert.", guess the 12: was for an old version
    #                     lr = self.args.learning_rate * 0.1
    #                 else:
    #                     lr = self.args.learning_rate
    #
    #                 if any(nd in key for nd in no_decay):
    #                     optimizer_grouped_parameters += [
    #                         {"params": [value], "lr": lr, "weight_decay": 0.01}
    #                     ]
    #
    #                 if not any(nd in key for nd in no_decay):
    #                     optimizer_grouped_parameters += [
    #                         {"params": [value], "lr": lr, "weight_decay": 0.0}
    #                     ]
    #         assert len(list(self.named_parameters())) == len(optimizer_grouped_parameters)
    #
    #     num_train_optimization_steps = (
    #             int(
    #                 len(self.train_dataloader())
    #                 / self.args.gradient_accumulation_steps
    #             )
    #             * self.args.num_train_epochs
    #     )
    #
    #     optimizer = BertAdam(
    #         optimizer_grouped_parameters,
    #         lr=self.args.learning_rate,
    #         warmup=self.args.warmup_proportion,
    #         t_total=num_train_optimization_steps,
    #     )
    #     return optimizer
    #     # return torch.optim.Adam(self.parameters(),lr=self.args.learning_rate)

    # def get_progress_bar_dict(self):
    #     tqdm_dict = super().get_progress_bar_dict()
    #     tqdm_dict['t'] = datetime.now(timezone('Europe/Brussels')).strftime("%X")
    #     return tqdm_dict


class FakeDataset:
    def __init__(self, size=2048, load_time=0.0005, nb_samples=999999):
        self.size, self.load_time, self.nb_samples = size, load_time, nb_samples


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        time.sleep(self.load_time)
        return torch.from_numpy(np.random.rand(self.size)), 1  # return img, label

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.nb_samples

def main_pl():
    # s = t()
    # from devlbert.devlbert import BertForMultiModalPreTraining, BertConfig
    # parser = jsonargparse.ArgumentParser()
    # parser = add_program_argparse_args(parser)
    #
    # # add all the available trainer options to argparse
    # # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    # # parser = Trainer.add_argparse_args(parser)
    # # parser = add_jsonargparse_args(Trainer, parser)
    #
    # parser.add_class_arguments(Trainer, 'trainer', as_group=True)
    # args = parser.parse_args()
    # if args.trainer.min_epochs != args.trainer.max_epochs:
    #     raise NotImplementedError
    # else:
    #     args.num_train_epochs = args.trainer.min_epochs
    # if not is_on_tpus():
    #     args.trainer.tpu_cores = None
    # else:
    #     if HOST == 'LIIR':
    #         if args.trainer.gpus == None:
    #             args.trainer.gpus = 4
    #     elif HOST == 'VSC':
    #         if args.trainer.gpus == None:
    #             args.trainer.gpus = 8
    # if args.debug:
    #     logger.setLevel(logging.DEBUG)
    # if args.baseline:
    #     raise NotImplementedError
    #
    # logger.info(yaml.dump(args))
    #
    # #region Getting BertConfig and tweaking it based on args
    # config = BertConfig.from_json_file(args.config_file)
    # if args.freeze > config.t_biattention_id[0]:
    #     config.fixed_t_layer = config.t_biattention_id[0]
    # if args.without_coattention:
    #     config.with_coattention = False
    # if args.predict_feature:
    #     config.v_target_size = 2048
    #     config.predict_feature = True
    # else:
    #     config.v_target_size = 1601
    #     config.predict_feature = False
    # #endregion
    #
    #
    # if args.gradient_accumulation_steps < 1:
    #     raise ValueError(
    #         "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
    #             args.gradient_accumulation_steps
    #         )
    #     )
    # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    #
    # # random.seed(args.seed)
    # # np.random.seed(args.seed)
    # # torch.manual_seed(args.seed)
    # # seed_everything(args.seed DONE IN PL ALREADY
    #
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # tokenizer = BertTokenizer.from_pretrained(
    #     args.bert_model, do_lower_case=args.do_lower_case
    # )
    #
    # class EmptyData(object):
    #
    #
    #     def __init__(
    #             self
    #     ):
    #         self.data = []
    #         self.num_dataset = 0
    #
    #     def __iter__(self):
    #
    #         for batch in self.data:
    #             yield batch
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    train_dataset = FakeDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=256)
    #
    #
    #
    # if args.from_pretrained:
    #     model = BertForMultiModalPreTraining.from_pretrained(args.from_pretrained, config, args=args)
    # else:
    #     model = BertForMultiModalPreTraining(config)
    #
    # if args.pt2_run:
    #     list_of_files = glob.glob(f'{args.output_dir}/epoch=*.ckpt')  # * means all if need specific format then *.csv
    #     latest_ckpt = max(list_of_files, key=os.path.getctime)
    #     model = BertForMultiModalPreTraining.load_from_checkpoint(latest_ckpt, config=config, args=args)
    #
    model = DummyModel()
    DUMMY_CHECKPOINT_DIR = 'dummy_ckpt_dir'
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=DUMMY_CHECKPOINT_DIR, save_top_k=-1)
    trainer = pl.Trainer(
        tpu_cores=8,
        callbacks=[lr_monitor, checkpoint_callback],
        profiler='simple')
    # print_gpu_mem()
    # myprint("***** Running training *****")
    # myprint("  Num examples =", train_dataset.num_dataset,
    #         "  Batch size =", args.train_batch_size,
    #         "  Num steps =", len(train_dataset))
    # myprint(f'Getting to trainer.fit took {t() - first_start} seconds')
    # trainer.tune(model,train_dataloader=train_dataset)
    trainer.fit(model, train_dataloader=train_dataloader)


if __name__ == "__main__":
    # main()
    main_pl()
