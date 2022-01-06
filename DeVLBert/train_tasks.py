#region pre-stuff
import argparse
import datetime
import json
import logging
import glob
import os
import yaml
import re
from constants import FINETUNE_DATA_ROOT_DIR, MINI_FT_EPOCHS
from pretorch_util import assign_visible_gpus
assign_visible_gpus()
from devlbert.myplmodule import get_core_module

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=yaml.YAMLLoadWarning)
import random
from io import open
import numpy as np
from time import time, sleep
# from tensorboardX import SummaryWriter # Nathan
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from bisect import bisect
from easydict import EasyDict as edict
import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.multiprocessing as mp

torch.multiprocessing.set_sharing_strategy('file_system')


from pytorch_pretrained_bert.optimization import WarmupLinearSchedule


# from parallel.parallel import DataParallelModel, DataParallelCriterion

from devlbert.task_utils import LoadDatasets, LoadLosses, ForwardModelsTrain, ForwardModelsVal
from devlbert.optimization import BertAdam, Adam, Adamax
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import devlbert.utils as utils
import torch.distributed as dist
from util import setup, cleanup, myprint
myprint("Post-torch imports train_tasks.py")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
#endregion
from cfg_train_tasks import FGS
OPTIMIZER_CKPT_PREFIX = "optimizer_state_"
LR_SCHEDULER_CKPT_PREFIX = "lr_scheduler_state_"
def main():
    #region parser stuff
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     "--bert_model",
    #     default="bert-base-uncased",
    #     type=str,
    #     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #     "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    # )
    # parser.add_argument(
    #     "--from_pretrained",
    #     default="bert-base-uncased",
    #     type=str,
    #     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #     "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     default="save",
    #     type=str,
    #     help="The output directory where the model checkpoints will be written.",
    # )
    # parser.add_argument(
    #     "--config_file",
    #     default="config/bert_config.json",
    #     type=str,
    #     help="The config file which specified the model details.",
    # )
    # parser.add_argument(
    #     "--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam."
    # )
    # parser.add_argument(
    #     "--num_train_epochs",
    #     default=20,
    #     type=int,
    #     help="Total number of training epochs to perform.",
    # )
    # parser.add_argument(
    #     "--warmup_proportion",
    #     default=0.1,
    #     type=float,
    #     help="Proportion of training to perform linear learning rate warmup for. "
    #     "E.g., 0.1 = 10%% of training.",
    # )
    # parser.add_argument(
    #     "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    # )
    # parser.add_argument(
    #     "--do_lower_case",
    #     default=True,
    #     type=bool,
    #     help="Whether to lower case the input text. True for uncased models, False for cased models.",
    # )
    # parser.add_argument(
    #     "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    # )
    # parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumualte before performing a backward/update pass.",
    # )
    # parser.add_argument(
    #     "--fp16",
    #     action="store_true",
    #     help="Whether to use 16-bit float precision instead of 32-bit",
    # )
    # parser.add_argument(
    #     "--loss_scale",
    #     type=float,
    #     default=0,
    #     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #     "0 (default value): dynamic loss scaling.\n"
    #     "Positive power of 2: static loss scaling value.\n",
    # )
    # parser.add_argument(
    #     "--num_workers", type=int, default=16, help="Number of workers in the dataloader."
    # )
    # parser.add_argument(
    #     "--save_name",
    #     default='',
    #     type=str,
    #     help="save name for training.",
    # )
    # parser.add_argument(
    #     "--use_chunk", default=0, type=float, help="whether use chunck for parallel training."
    # )
    # parser.add_argument(
    #     "--in_memory", default=False, type=bool, help="whether use chunck for parallel training."
    # )
    # parser.add_argument(
    #     "--optimizer", default='BertAdam', type=str, help="whether use chunck for parallel training."
    # )
    # parser.add_argument(
    #     "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    # )
    # parser.add_argument(
    #     "--freeze", default = -1, type=int,
    #     help="till which layer of textual stream of vilbert need to fixed."
    # )
    # parser.add_argument(
    #     "--vision_scratch", action="store_true", help="whether pre-trained the image or not."
    # )
    # parser.add_argument(
    #     "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
    # )
    # parser.add_argument(
    #     "--lr_scheduler", default='mannul', type=str, help="whether use learning rate scheduler."
    # )
    # parser.add_argument(
    #     "--baseline", action="store_true", help="whether use single stream baseline."
    # )
    # parser.add_argument(
    #     "--compact", action="store_true", help="whether use compact vilbert model."
    # )
    # parser.add_argument(
    #     "--use_ema", action="store_true", help="whether to use EMA."
    # )
    # parser.add_argument(
    #     "--ema_decay_ratio", type=float, default=0.9999, help='EMA dacay ratio.'
    # )
    # parser.add_argument(
    #     "--world_size",
    #     type=int,
    #     default=len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")),
    #     help="Number of processes, should equal number of GPUs you intend to use.",
    # )
    # args = parser.parse_args()
    args = argparse.Namespace()
    args.__dict__ = {k: v.value for k, v in FGS.__flags.items()}
    #endregion
    assert not args.world_size > len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")), "World size bigger than number of available GPUs"
    mp.spawn(main_single_process,
         args=(args,),
         nprocs=args.world_size,
         join=True)

def main_single_process(rank, args):
    myprint("Entered main single process")
    args.local_rank = rank

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        setup(rank=rank, world_size=args.world_size,tasks=args.tasks)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend="nccl")

    if args.mini:
        print(f"Mini run: setting # train epochs to {MINI_FT_EPOCHS}")
        args.num_train_epochs = MINI_FT_EPOCHS
    myprint("Entered single process")
    with open('devlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.load(f))
    #Nathan: different root dirs on different servers
    for t in task_cfg:
        for k, v in task_cfg[t].items():
            if '__ROOT__' in str(v):
                task_cfg[t][k] = v.replace('__ROOT__', FINETUNE_DATA_ROOT_DIR)
    #Nathan
    for tsk in ('TASK0','TASK3'):
        task_cfg[tsk]['features_h5path1'] = task_cfg[tsk]['train_features_h5path1']

    task_cfg['TASK3']['val_annotations_jsonpath'] = task_cfg['TASK3']['train_val_annotations_jsonpath']

    if args.mini:
        for p in ('train_annotations_jsonpath','val_annotations_jsonpath'):
            split_path = os.path.split(task_cfg['TASK3'][p])
            task_cfg['TASK3'][p] = os.path.join(*[split_path[0], 'mini_' + split_path[1]])

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
    # print(task_ids) #['TASK0'] ['TASK3']
    logdir = os.path.join(args.output_dir,'logs',timeStamp)

    try:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    except Exception:
        pass

    tbLogger = utils.tbLogger(logdir, savePath, task_names, task_ids, task_num_iters, args.gradient_accumulation_steps,save_logger=True)

    # if n_gpu > 0:
        # torch.cuda.manual_seed_all(args.seed)

    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    except Exception:
        pass

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
    model.to(device) # Nathan
    if args.local_rank != -1:
        try:
            # from apex.parallel import DistributedDataParallel as DDP # Nathan
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        # model = DDP(model, delay_allreduce=True)
        myprint(device,args.local_rank)
        model = DDP(model,find_unused_parameters=True,device_ids=[rank],output_device=rank) # Nathan based on https://github.com/NVIDIA/apex/issues/539

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
    s = None
    ETR = "?"

    # Nathan: to resume by default from checkpoints present in the output_dir TODO this didn't work: the validation loss when resuming jumped a lot, so it seems it didn't resume correctly
    start_epoch = 0
    maybe_ema = "_ema" if args.use_ema else ""
    path_search_string = os.path.join(savePath,f"pytorch_model_*{maybe_ema}.bin")
    paths = glob.glob(path_search_string)
    print(f"Paths found with existing checkpoints: {paths}")
    # if len(paths) != 0:
    #     latest_model_state_dict_file = max(paths, key=os.path.getctime)
    #     print(f"Found existing checkpoint {latest_model_state_dict_file} in output_dir. Resuming from that checkpoint")
    #     ckpt_epoch_number = int(re.match(".*pytorch_model_([0-9]+)_ema.bin",latest_model_state_dict_file).groups()[0])
    #     start_epoch = 1 + ckpt_epoch_number
    #     latest_optimizer_state_dict_file = os.path.join(savePath,f"{OPTIMIZER_CKPT_PREFIX}{ckpt_epoch_number}.bin")
    #     latest_lr_scheduler_state_dict_file = os.path.join(savePath,f"{LR_SCHEDULER_CKPT_PREFIX}{ckpt_epoch_number}.bin")
    #
    #     ckpt_state_dict = torch.load(latest_model_state_dict_file,map_location=torch.device('cpu'))
    #     optimizer_state_dict = torch.load(latest_optimizer_state_dict_file,map_location=torch.device('cpu'))
    #     lr_scheduler_state_dict = torch.load(latest_lr_scheduler_state_dict_file,map_location=torch.device('cpu'))
    #
    #     core_module = get_core_module(model)
    #     core_module.load_state_dict(ckpt_state_dict)
    #     optimizer.load_state_dict(optimizer_state_dict)
    #     lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    # for epochId in range(args.num_train_epochs):
    for epochId in range(start_epoch,args.num_train_epochs):
        remaining_epochs = args.num_train_epochs - epochId
        print(f"Remaining epochs: {remaining_epochs}")
        if s is not None:
            last_epoch_duration = time() - s
            ETR = str(datetime.timedelta(seconds=int(remaining_epochs * last_epoch_duration)))
        s = time()
        if default_gpu:
            myprint(
                f"EPOCH {epochId} out of {args.num_train_epochs}",
                "|","*" * epochId," " * remaining_epochs, "|",
                f"ETR: {ETR} ")


        model.train()
        
        # print("="*50,"SKIPPING TRAINING FOR DEBUGGING","="*50)
        # max_num_iter = 0
        iterator = range(max_num_iter)
        if default_gpu:
            iterator = tqdm(iterator, desc="Step", position=0)
        for step in iterator:
            iterId = startIterID + step + (epochId * max_num_iter)
            for task_id in task_ids:
                if iterId >= task_start_iter[task_id]:
                # if iterId % task_interval[task_id] == 0:
                #     if not default_gpu:
                #         myprint(f"SLEEPING FOR MP DEBUGGING PURPOSE: {args.local_rank}")
                #         sleep(10e5)
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

            if step % (200 * args.gradient_accumulation_steps) == 0 and step != 0 and default_gpu:
                tbLogger.showLossTrain()


        #region post-iteration stuff
        model.eval()
        # If EMA is used, we use averaged model to make eval
        if args.use_ema:
            # backup current params on cpu
            bkp_state_dict = {param_name: param_tensor.cpu().detach() for param_name, param_tensor in model.state_dict().items()}
            # load averaged params
            model.load_state_dict(ema_state_dict)

        if default_gpu:
            # Save a trained model
            output_model_file = os.path.join(savePath, "pytorch_model_" + str(epochId) + ".bin")
            logger.info("** ** * Saving fine - tuned model on " + timeStamp + f" in {output_model_file} ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            try:
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
            except Exception:
                pass
            torch.save(model_to_save.state_dict(), output_model_file)
            # Nathan: also save optimizer state and lr_scheduler state
            output_optimizer_file = os.path.join(savePath, OPTIMIZER_CKPT_PREFIX + str(epochId) + ".bin")
            output_lr_scheduler_file = os.path.join(savePath, LR_SCHEDULER_CKPT_PREFIX + str(epochId) + ".bin")

            torch.save(optimizer.state_dict(),output_optimizer_file)
            torch.save(lr_scheduler.state_dict(), output_lr_scheduler_file)
            # If EMA is used, save averaged model
            if args.use_ema:
                output_ema_state_dict = {}
                for param_name in model.state_dict():
                    assert param_name in ema_state_dict
                    if hasattr(model, "module"):
                        output_ema_state_dict[param_name[7:]] = ema_state_dict[param_name] # skip prefix "module."
                    else:
                        output_ema_state_dict[param_name] = ema_state_dict[param_name]
                output_ema_model_file = os.path.join(savePath, "pytorch_model_" + str(epochId) + "_ema.bin")
                torch.save(output_ema_state_dict, output_ema_model_file)
        # when run evaluate, we run each task sequentially.
        for task_id in task_ids:
            try:
                for i, batch in enumerate(task_dataloader_val[task_id]):
                    loss, score, batch_size = ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses)
                    tbLogger.step_val(epochId, float(loss), float(score), task_id, batch_size, 'val')
                    if default_gpu:
                        sys.stdout.write('%d/%d\r' % (i, len(task_dataloader_val[task_id])))
                        sys.stdout.flush()
            except ValueError:
                print(5)

        # If EMA is used, recover unaveraged params
        if args.use_ema:
            model.load_state_dict(bkp_state_dict)

        ave_score = tbLogger.showLossVal()
        if args.lr_scheduler == 'automatic':
            lr_scheduler.step(ave_score)
            logger.info("best average score is %3f" %lr_scheduler.best)
        else:
            lr_scheduler.step()

        #endregion


    tbLogger.txt_close()
    cleanup()
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()