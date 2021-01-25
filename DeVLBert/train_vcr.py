import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict

import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_pretrained_bert.optimization import WarmupLinearSchedule

# from parallel.parallel import DataParallelModel, DataParallelCriterion

from devlbert.task_utils import LoadDatasets, LoadLosses, ForwardModelsTrain
from devlbert.optimization import BertAdam, Adam, Adamax
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = batch
    batch_size = features.size(0)

    if task_id in ['TASK1', 'TASK2', 'TASK5', 'TASK6', 'TASK7']:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.unsqueeze(1).expand(batch_size, num_options, max_num_bbox, 2048).contiguous().view(-1, max_num_bbox, 2048)
        spatials = spatials.unsqueeze(1).expand(batch_size, num_options, max_num_bbox, 5).contiguous().view(-1, max_num_bbox, 5)
        image_mask = image_mask.unsqueeze(1).expand(batch_size, num_options, max_num_bbox).contiguous().view(-1, max_num_bbox)
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

    with torch.no_grad():
        vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit = \
                                            model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)

    vil_logit = vil_logit.view(batch_size, num_options)
    loss = task_losses[task_id](vil_logit, target)
    _, preds = torch.max(vil_logit, 1)
    batch_score = list((preds == target).cpu().numpy())

    return float(loss), batch_score, batch_size

class TbLogger(object):
    def __init__(self, log_dir, txt_dir, task_names, task_ids, task_num_iters, gradient_accumulation_steps, save_logger=False, txt_name='out.txt'):
        logger.info("logging file at: " + log_dir)

        self.save_logger=save_logger
        if self.save_logger:
            self.logger = SummaryWriter(log_dir=log_dir)

        self.txt_f = open(txt_dir + '/' + txt_name, 'w')
        self.task_id2name = {ids:name.replace('+', 'plus') for ids, name in zip(task_ids, task_names)}
        self.task_ids = task_ids
        self.task_loss = {task_id:0 for task_id in task_ids}
        self.task_loss_tmp = {task_id:0 for task_id in task_ids}
        self.task_score_tmp = {task_id:0 for task_id in task_ids}
        self.task_norm_tmp = {task_id:0 for task_id in task_ids}
        self.task_step = {task_id:0 for task_id in task_ids}
        self.task_step_tmp = {task_id:0 for task_id in task_ids}
        self.task_num_iters = task_num_iters
        self.epochId = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.task_loss_val = {task_id:0 for task_id in task_ids}
        self.task_score_val = {task_id:[] for task_id in task_ids}
        self.task_step_val = {task_id:0 for task_id in task_ids}
        self.task_datasize_val = {task_id:0 for task_id in task_ids}

    def txt_close(self):
        self.txt_f.close()

    def linePlot(self, step, val, split, key, xlabel="None"):
        if self.save_logger:
            self.logger.add_scalar(split + "/" + key, val, step)

    def step_train(self, epochId, stepId, loss, score, norm, task_id, split):
        self.task_loss[task_id] += loss
        self.task_loss_tmp[task_id] += loss
        self.task_score_tmp[task_id] += score
        self.task_norm_tmp[task_id] += norm
        self.task_step[task_id] += self.gradient_accumulation_steps
        self.task_step_tmp[task_id] += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(stepId, loss, split, self.task_id2name[task_id] + '_loss')
        self.linePlot(stepId, score, split, self.task_id2name[task_id] + '_score')

    def step_val(self, epochId, loss, score, task_id, batch_size, split):
        self.task_loss_val[task_id] += loss
        self.task_score_val[task_id] += score
        self.task_step_val[task_id] += self.gradient_accumulation_steps
        self.task_datasize_val[task_id] += batch_size

    def showLossVal(self):
        progressInfo = "Eval Ep: %d " %self.epochId
        lossInfo = 'Validation '
        ave_score = 0
        ave_loss = 0
        for task_id in self.task_ids:
            loss = self.task_loss_val[task_id] / float(self.task_step_val[task_id])
            score = sum(self.task_score_val[task_id]) / float(self.task_datasize_val[task_id])
            ave_score += score
            ave_loss += loss
            lossInfo += '[%s]: loss %.3f score %.3f ' %(self.task_id2name[task_id], loss, score * 100.0)

            self.linePlot(self.epochId, loss, 'val', self.task_id2name[task_id] + '_loss')
            self.linePlot(self.epochId, score, 'val', self.task_id2name[task_id] + '_score')

        q2ar = 0
        lis1 = self.task_score_val["TASK1"]; lis2 = self.task_score_val["TASK2"]
        for q2a, qa2r in zip(lis1, lis2):
            if q2a == 1 and qa2r == 1: q2ar += 1
        q2ar = q2ar / float(self.task_datasize_val["TASK1"])
        lossInfo += '[VCR_Q-AR]: score %.3f ' % (q2ar * 100.0)

        ave_score = ave_score / len(self.task_ids)
        self.task_loss_val = {task_id:0 for task_id in self.task_loss_val}
        # self.task_score_val = {task_id:[] for task_id in self.task_score_val}
        self.task_score_val["TASK1"].clear(); self.task_score_val["TASK2"].clear()
        self.task_datasize_val = {task_id:0 for task_id in self.task_datasize_val}
        self.task_step_val = {task_id:0 for task_id in self.task_ids}
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return ave_score

    def showLossTrain(self):
        # show the current loss, once showed, reset the loss.
        lossInfo = ''
        for task_id in self.task_ids:
            if self.task_num_iters[task_id] > 0:
                if self.task_step_tmp[task_id]:
                    lossInfo += '[%s]: iter %d Ep: %.2f loss %.3f score %.3f lr %.6g ' %(self.task_id2name[task_id], \
                        self.task_step[task_id], self.task_step[task_id] / float(self.task_num_iters[task_id]), \
                                            self.task_loss_tmp[task_id] / float(self.task_step_tmp[task_id]), \
                                            self.task_score_tmp[task_id] / float(self.task_step_tmp[task_id]), \
                                            self.task_norm_tmp[task_id] / float(self.task_step_tmp[task_id]))

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = {task_id:0 for task_id in self.task_ids}
        self.task_loss_tmp = {task_id:0 for task_id in self.task_ids}
        self.task_score_tmp =  {task_id:0 for task_id in self.task_ids}
        self.task_norm_tmp = {task_id:0 for task_id in self.task_ids}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.", 
    )
    parser.add_argument(
        "--use_chunk", default=0, type=float, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--in_memory", default=False, type=bool, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--optimizer", default='BertAdam', type=str, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--freeze", default = -1, type=int, 
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument(
        "--vision_scratch", action="store_true", help="whether pre-trained the image or not."
    )
    parser.add_argument(
        "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
    )
    parser.add_argument(
        "--lr_scheduler", default='mannul', type=str, help="whether use learning rate scheduler."
    )  
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--compact", action="store_true", help="whether use compact vilbert model."
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="whether to use EMA."
    )
    parser.add_argument(
        "--ema_decay_ratio", type=float, default=0.9999, help='EMA dacay ratio.'
    )
    args = parser.parse_args()
    with open('devlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.load(f))

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from devlbert.basebert import BaseBertForVLTasks
    elif args.compact:
        from devlbert.vilbert_compact import BertConfig
        from devlbert.vilbert_compact import VILBertForVLTasks
    else:
        from devlbert.devlbert import BertConfig
        from devlbert.devlbert import DeVLBertForVLTasks

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)
        task_lr.append(task_cfg[task]['lr'])

    base_lr = min(task_lr)
    loss_scale = {}
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        loss_scale[task] = task_lr[i] / base_lr

    if args.save_name:
        prefix = '-' + args.save_name
    else:
        prefix = ''
    timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0] + prefix
    savePath = os.path.join(args.output_dir, timeStamp)

    bert_weight_name = json.load(open("config/" + "bert-base-uncased_weight_name.json", "r"))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    try:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
    except Exception:
        pass

    config = BertConfig.from_json_file(args.config_file)
    if default_gpu:
        # save all the hidden parameters. 
        with open(os.path.join(savePath, 'command.txt'), 'w') as f:
            print(args, file=f)  # Python 3.x
            print('\n', file=f)
            print(config, file=f)

    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, \
            task_dataloader_train, task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split('-'))

    tbLogger = TbLogger(timeStamp, savePath, task_names, task_ids, task_num_iters, args.gradient_accumulation_steps)

    # if n_gpu > 0:
        # torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_optimization_steps = max(task_num_iters.values()) * args.num_train_epochs // args.gradient_accumulation_steps
    num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()])

    task_start_iter = {}
    task_interval = {}
    for task_id, num_iter in task_num_iters.items():
        task_start_iter[task_id] = num_train_optimization_steps - (task_cfg[task]['num_epoch'] * num_iter // args.gradient_accumulation_steps)
        task_interval[task_id] = num_train_optimization_steps // (task_cfg[task]['num_epoch'] * num_iter // args.gradient_accumulation_steps)

    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained, config, num_labels=num_labels, default_gpu=default_gpu
            )
    else:
        model = DeVLBertForVLTasks.from_pretrained(
            args.from_pretrained, config, num_labels=num_labels, default_gpu=default_gpu
            )

    task_losses = LoadLosses(args, task_cfg, args.tasks.split('-'))
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if 'embeddings' in name:
                bert_weight_name_filtered.append(name)
            elif 'encoder' in name:
                layer_num = name.split('.')[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)
        
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False
            
        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    optimizer_grouped_parameters = []
    lr = args.learning_rate
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'vil_prediction' in key:
                # if args.learning_rate <= 2e-5:
                lr = 1e-4
            else:
                if args.vision_scratch:
                    if key[12:] in bert_weight_name:
                        lr = args.learning_rate
                    else:
                        lr = 1e-4
                else:
                    lr = args.learning_rate
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    max_num_iter = max(task_num_iters.values())
    max_batch_size = max(task_batch_size.values())
    
    if args.optimizer == 'BertAdam':
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps,
            schedule='warmup_constant',
        )
    elif args.optimizer == 'Adam':
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=base_lr,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps,
            schedule='warmup_constant',
        )
    elif args.optimizer == 'Adamax':
        optimizer = Adamax(
            optimizer_grouped_parameters,
            lr=base_lr,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps,
            schedule='warmup_constant',
        )        

    if args.lr_scheduler == 'automatic':
        lr_scheduler = ReduceLROnPlateau(optimizer, \
                        mode='max',
                        factor=0.2, 
                        patience=1, 
                        cooldown=1,
                        threshold=0.001)
    elif args.lr_scheduler == 'mannul':
        lr_reduce_list = np.array([12, 16])
        # lr_reduce_list = np.array([6, 8, 10])        
        def lr_lambda_fun(epoch):
            return pow(0.1, np.sum(lr_reduce_list <= epoch))
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", task_batch_size)
        print("  Num steps: %d" %num_train_optimization_steps)

    if args.use_ema:
        ema_state_dict = {}
        for param_name, param_tensor in model.state_dict().items():
            ema_state_dict[param_name] = param_tensor.clone().detach() # we currently store the ema params on GPU

    startIterID = 0
    # initialize the data iteration.
    task_iter_train = {name:None for name in task_ids}
    task_count = {name:0 for name in task_ids}
    for epochId in range(args.num_train_epochs):
        model.train()
        for step in range(max_num_iter):
            iterId = startIterID + step + (epochId * max_num_iter)
            for task_id in task_ids:
                if iterId >= task_start_iter[task_id]:
                # if iterId % task_interval[task_id] == 0:
                    loss, score = ForwardModelsTrain(args, task_cfg, device, task_id, task_count, task_iter_train, task_dataloader_train, model, task_losses, task_start_iter)
                    loss = loss * loss_scale[task_id]
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps 

                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        model.zero_grad()
                        if args.use_ema:
                            for param_name, param_tensor in model.state_dict().items():
                                assert param_name in ema_state_dict
                                ema_state_dict[param_name] -= (1.0 - args.ema_decay_ratio) * (ema_state_dict[param_name] - param_tensor)

                        if default_gpu:
                            tbLogger.step_train(epochId, iterId, float(loss), float(score), optimizer.show_lr(), task_id, 'train')

            if step % (20 * args.gradient_accumulation_steps) == 0 and step != 0 and default_gpu:
                tbLogger.showLossTrain()

        model.eval()
        # If EMA is used, we use averaged model to make eval
        if args.use_ema:
            # backup current params on cpu
            bkp_state_dict = {param_name: param_tensor.cpu().detach() for param_name, param_tensor in model.state_dict().items()}
            # load averaged params
            model.load_state_dict(ema_state_dict)
                
        # when run evaluate, we run each task sequentially. 
        for task_id in task_ids:
            for i, batch in enumerate(task_dataloader_val[task_id]):
                loss, score, batch_size = ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses)
                tbLogger.step_val(epochId, float(loss), score, task_id, batch_size, 'val')
                # if default_gpu:
                #     print(score)
                #     sys.stdout.write('%d/%d\r' % (i, len(task_dataloader_val[task_id])))
                #     sys.stdout.flush()

        # If EMA is used, recover unaveraged params
        if args.use_ema:
            model.load_state_dict(bkp_state_dict)

        ave_score = tbLogger.showLossVal()
        if args.lr_scheduler == 'automatic':
            lr_scheduler.step(ave_score)
            logger.info("best average score is %3f" %lr_scheduler.best)
        else:
            lr_scheduler.step()

        # if default_gpu:
        #     # Save a trained model
        #     logger.info("** ** * Saving fine - tuned model on " + timeStamp + "** ** * ")
        #     model_to_save = (
        #         model.module if hasattr(model, "module") else model
        #     )  # Only save the model it-self
        #
        #     if not os.path.exists(savePath):
        #         os.makedirs(savePath)
        #     output_model_file = os.path.join(savePath, "pytorch_model_" + str(epochId) + ".bin")
        #     torch.save(model_to_save.state_dict(), output_model_file)
        #     # If EMA is used, save averaged model
        #     if args.use_ema:
        #         output_ema_state_dict = {}
        #         for param_name in model.state_dict():
        #             assert param_name in ema_state_dict
        #             if hasattr(model, "module"):
        #                 output_ema_state_dict[param_name[7:]] = ema_state_dict[param_name] # skip prefix "module."
        #             else:
        #                 output_ema_state_dict[param_name] = ema_state_dict[param_name]
        #         output_ema_model_file = os.path.join(savePath, "pytorch_model_" + str(epochId) + "_ema.bin")
        #         torch.save(output_ema_state_dict, output_ema_model_file)

    tbLogger.txt_close()
    
if __name__ == "__main__":

    main()