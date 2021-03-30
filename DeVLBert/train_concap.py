from time import time as t, sleep
import glob

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
import json
import logging
import os
import random
from io import open
import math
import sys
from time import strftime
from datetime import datetime
from pathlib import Path

import torch.multiprocessing as mp
START = t()
from timeit import default_timer as timer
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from devlbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from devlbert.devlbert import BertForMultiModalPreTraining, BertConfig
import torch.distributed as dist
import pdb
from time import gmtime
from constants import MODEL_CKPT_DIR
from util import rank_to_device, MyLogger, myprint
from torch.nn.parallel import DistributedDataParallel as DDP
logging.setLoggerClass(MyLogger)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

last_checkpoint_time = t()
MASTER_RANK = 0

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main():
    # region Parser stuff
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--train_file",
        default="data/training",
        # default="data/conceptual_caption/training",
        type=str,
        # required=True,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--validation_file",
        default="data/validation",
        # default="data/conceptual_caption/validation",
        type=str,
        # required=True,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default=MODEL_CKPT_DIR,
        type=str,
        # required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=36,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
             "Sequences longer than this will be truncated, and sequences shorter \n"
             "than this will be padded.",
    )
    parser.add_argument("--predict_feature", action="store_true", help="visual target.")
    parser.add_argument(
        "--train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=12.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
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
        "--img_weight", default=1, type=float, help="weight for image loss"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--on_memory",
        action="store_true",
        help="Whether to load train samples into memory or use disk",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
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
        "--num_workers",
        type=int,
        default=3,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Number of processes, should equal number of GPUs you intend to use.",
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Wheter to use the baseline model (single bert)."
    )
    parser.add_argument(
        "--freeze", default=-1, type=int,
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument(
        "--use_chuncks", default=0, type=float, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--distributed", action="store_true", help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--without_coattention", action="store_true", help="whether pair loss."
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="if we need to continue a stopped pretraining procedure, add this"
    )
    parser.add_argument(
        "--gpus",
        nargs='+', type=int,
        help="which gpus to consider"
    )
    parser.add_argument(
        "--mini", action="store_true", help="Whether to train on mini data, just to test the whole training loop"
    )
    parser.add_argument(
        "--checkpoint_period",
        type=int,
        default=60 * 60,
        help="Number seconds between each time a checkpoint is stored",
    )
    parser.add_argument(
        "--debug", action="store_true", help="If true sets logging level to debug"
    )
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in args.gpus])
    # endregion

    if args.save_name is not '':
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a")
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))
    savePath = os.path.join(args.output_dir, timeStamp)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    mp.spawn(main_single_process,
             args=(args, savePath),
             nprocs=args.world_size,
             join=True)


def main_single_process(rank, args, savePath):
    setup(rank,args.world_size)
    s = t()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from devlbert.basebert import BertForMultiModalPreTraining
    else:
        from devlbert.devlbert import BertForMultiModalPreTraining, BertConfig
    logger.info('\r\n'.join(f'{k}: {v}' for k, v in args.__dict__.items()))
    config = BertConfig.from_json_file(args.config_file)
    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]
    if args.without_coattention:
        config.with_coattention = False
    # save all the hidden parameters.
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)
    bert_weight_name = json.load(open("config/" + "bert-base-uncased_weight_name.json", "r"))

    device = torch.device("cuda", rank_to_device(rank))
    torch.cuda.set_device(device)
    n_gpu = 1
    default_gpu = rank == 0

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    train_dataset = ConceptCapLoaderTrain(
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=args.num_workers,
        savePath=savePath,
        mini=args.mini
    )
    # validation_dataset = ConceptCapLoaderVal(
    #     args.validation_file,
    #     tokenizer,
    #     seq_len=args.max_seq_length,
    #     batch_size=args.train_batch_size,
    #     predict_feature=args.predict_feature,
    #     num_workers=2,
    #     distributed=args.distributed,
    # )
    num_train_optimization_steps = (
            int(
                train_dataset.num_dataset
                / args.train_batch_size
                / args.gradient_accumulation_steps
            )
            * args.num_train_epochs
    )

    viz = TBlogger(Path(args.output_dir, "logs"), Path(savePath).name)

    # pdb.set_trace()
    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False
    if args.from_pretrained:
        model = DDP(BertForMultiModalPreTraining.from_pretrained(args.from_pretrained, config).to(device),
                    device_ids=[device])
        ckpt_load_path = Path(savePath, f"model.checkpoint")
        if args.continue_training and Path(ckpt_load_path).exists():
            # ckpt_load_path = os.path.join(args.from_pretrained,
            #                               "pytorch_model_{}.bin".format(int(args.start_epoch) - 1))
            # model = BertForMultiModalPreTraining.from_pretrained(ckpt_load_path, config)
            print(f"Loading model from checkpoint {ckpt_load_path}")
            # map_location = {f'cuda:{rank_to_device(MASTER_RANK)}': str(device)}
            model.load_state_dict(
                torch.load(ckpt_load_path, map_location=device))
            torch.cuda.empty_cache()
            # model = torch.load(ckpt_load_path, map_location=map_location)

            # if not default_gpu:
            #     print(5)
            # print("SLEEPING FOR DEBUGGING")
            # sleep(10000000)

            # Load correct epoch
            epoch_file = get_epoch_file_path(savePath)
            if epoch_file is not None:
                with open(epoch_file, 'r') as f:
                    args.start_epoch = json.load(f)
                print(f"Loading start epoch from {epoch_file}")
    else:
        model = DDP(BertForMultiModalPreTraining(config).to(device), device_ids=[device])

    if args.fp16:
        model.half()

    # elif n_gpu > 1:
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
    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
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
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                # if key[12:] in bert_weight_name:
                if key[5:] in bert_weight_name:  # Nathan: starts with "bert.", guess the 12: was for an old version
                    lr = args.learning_rate * 0.1
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
    # set different parameters for vision branch and lanugage branch.
    if args.fp16:
        try:
            from apex.contrib.optimizers import FP16_Optimizer
            from apex.contrib.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        if args.from_pretrained:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,

            )
        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,
            )
        if args.continue_training:
            opt_state_dict_path = os.path.join(
                savePath, f"optimizer_state.checkpoint"
            )
            if Path(opt_state_dict_path).exists():
                print(f"Loading optimizer state dict from {opt_state_dict_path}")
                optimizer.load_state_dict(torch.load(opt_state_dict_path, map_location='cpu'))
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    global_step = 0
    myprint(f'Getting to epoch loop took {t() - s} seconds')
    # random.seed(torch.distributed.get_rank() * 2000)
    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):

        ## Store epoch idx
        # delete the old file
        old_path = get_epoch_file_path(savePath)
        if old_path is not None:
            os.remove(old_path)
        # store new file
        epoch_file = Path(savePath, f'start_epoch_{epochId}.json')
        with open(epoch_file, 'w') as f:
            json.dump(epochId, f)
            print(f"Storing done epochs in {epoch_file}")

        # Getting progress bar iterator for main process
        basic_iterator = enumerate(train_dataset, 1)
        iterator = tqdm(basic_iterator,
                        total=len(train_dataset),
                        initial=sum(train_dataset.get_core_ds().start_keys) // args.train_batch_size) \
            if default_gpu else basic_iterator

        s = t()
        first = True
        for step, batch in iterator:
            first = False
            if first:
                myprint(f"Loading first batch took {t() - s} seconds")
            else:
                myprint(f"Loading non-first batch took {t() - s} seconds")
            s = t()
            training_step(args, batch, default_gpu, device,
                          epochId, model, optimizer, rank, savePath,
                          step, train_dataset, viz)
            myprint(f"Entire train step took {t() - s} seconds")
            s = t()
        # Do the evaluation
        # torch.set_grad_enabled(False)
        # start_t = timer()
        # numBatches = len(validation_dataset)
        # eval_masked_loss_t = 0
        # eval_masked_loss_v = 0
        # eval_next_sentence_loss = 0
        # eval_total_loss = 0
        # eval_causal_loss = 0
        #
        # model.eval()
        # for step, batch in enumerate(validation_dataset, 1):
        #     batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        #
        #     input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, image_ids = (
        #         batch
        #     )
        #
        #     masked_loss_t, masked_loss_v, next_sentence_loss, causal_loss = model(
        #         input_ids,
        #         image_feat,
        #         image_loc,
        #         segment_ids,
        #         input_mask,
        #         image_mask,
        #         lm_label_ids,
        #         image_label,
        #         image_target,
        #         is_next,
        #     )
        #
        #     masked_loss_v = masked_loss_v * args.img_weight
        #     loss = masked_loss_t + masked_loss_v + next_sentence_loss + causal_loss
        #
        #     if n_gpu > 1:
        #         loss = loss.mean()  # mean() to average on multi-gpu.
        #         masked_loss_t = masked_loss_t.mean()
        #         masked_loss_v = masked_loss_v.mean()
        #         next_sentence_loss = next_sentence_loss.mean()
        #         causal_loss = causal_loss.mean()
        #
        #     eval_masked_loss_t += masked_loss_t.item()
        #     eval_masked_loss_v += masked_loss_v.item()
        #     eval_next_sentence_loss += next_sentence_loss.item()
        #     eval_total_loss += loss.item()
        #     eval_causal_loss += causal_loss.item()
        #
        #     end_t = timer()
        #     delta_t = " Time: %5.2fs" % (end_t - start_t)
        #     start_t = end_t
        #     progressString = "\r Evaluating split '%s' [%d/%d]\t" + delta_t
        #     sys.stdout.write(progressString % ('val', step, numBatches))
        #     sys.stdout.flush()
        #
        # eval_masked_loss_t = eval_masked_loss_t / float(numBatches)
        # eval_masked_loss_v = eval_masked_loss_v / float(numBatches)
        # eval_next_sentence_loss = eval_next_sentence_loss / float(numBatches)
        # eval_total_loss = eval_total_loss / float(numBatches)
        # eval_causal_loss = eval_causal_loss / float(numBatches)
        #
        # printFormat = "Evaluation: [Loss: %.5g][Loss_v: %.5g][Loss_t: %.5g][Loss_n: %.5g][Loss_c: %.5g]"
        # printInfo = [
        #     eval_total_loss,
        #     eval_masked_loss_v,
        #     eval_masked_loss_t,
        #     eval_next_sentence_loss,
        #     eval_causal_loss,
        # ]
        #
        # print(printFormat % tuple(printInfo))
        # torch.set_grad_enabled(True)

        # if default_gpu:
        #     viz.linePlot(epochId, eval_total_loss, "loss", "val")
        #     viz.linePlot(epochId, eval_masked_loss_t, "masked_loss_t", "val")
        #     viz.linePlot(epochId, eval_masked_loss_v, "masked_loss_v", "val")
        #     viz.linePlot(epochId, eval_next_sentence_loss, "next_sentence_loss", "val")
        #     viz.linePlot(epochId, eval_causal_loss, "causal_loss", "val")

        # Reset dataflow (esp. the pickled key index) after going through dataset
        train_dataset.reset_index()

        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, f"pytorch_model_{str(epochId)}.bin"
            )
            torch.save(model_to_save.state_dict(), output_model_file)
            # output_opt_state_dict_file = os.path.join(
            #     savePath, "optimizer_state_" + str(epochId) + ".bin"
            # torch.save(optimizer.state_dict(), output_opt_state_dict_file)
            # )
    cleanup()

def training_step(args, batch, default_gpu, device, epochId, model, optimizer, rank, savePath, step, train_dataset, viz):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, \
    image_ids, causal_label_t, causal_label_v = (
        batch
    )
    s = t()
    myprint(f'Doing Model input --> model output losses')
    # myprint(str(list(model.module.parameters())[0].device))
    # myprint(str(input_ids.device))
    # skip_sleep = False
    # if not default_gpu:
    #     print(5)
    # if not skip_sleep:
    #     print("SLEEPING FOR DEBUGGING")
    #     sleep(10000000)
    masked_loss_t, masked_loss_v, next_sentence_loss, \
    causal_prediction_v_loss, causal_prediction_t_loss, \
    causal_prediction_v2t_loss, causal_prediction_t2v_loss = model(
        input_ids,
        image_feat,
        image_loc,
        segment_ids,
        input_mask,
        image_mask,
        lm_label_ids,
        image_label,
        image_target,
        is_next,
        causal_label_t,
        causal_label_v
    )
    myprint(f'Model input --> model output losses took {t() - s}')

    masked_loss_v = masked_loss_v * args.img_weight
    loss = masked_loss_t + masked_loss_v + next_sentence_loss + \
           causal_prediction_v_loss + causal_prediction_t_loss + \
           causal_prediction_v2t_loss + causal_prediction_t2v_loss
    s = t()
    if args.fp16:
        optimizer.backward(loss)
    else:
        loss.backward()
    myprint(f'Calculating gradients took {t() - s}')
    if math.isnan(loss.item()):
        pdb.set_trace()
    if default_gpu:
        s = t()
        iterId = (sum(train_dataset.get_core_ds().current_key_idx_list) // len(batch)) + (epochId * len(train_dataset))
        viz.linePlot(iterId, loss.item(), "loss", "train")
        viz.linePlot(iterId, masked_loss_t.item(), "masked_loss_t", "train")
        viz.linePlot(iterId, masked_loss_v.item(), "masked_loss_v", "train")
        viz.linePlot(iterId, next_sentence_loss.item(), "next_sentence_loss", "train")
        viz.linePlot(iterId, causal_prediction_v_loss.item(), "causal_prediction_v_loss", "train")
        viz.linePlot(iterId, causal_prediction_t_loss.item(), "causal_prediction_t_loss", "train")
        viz.linePlot(iterId, causal_prediction_v2t_loss.item(), "causal_prediction_v2t_loss", "train")
        viz.linePlot(iterId, causal_prediction_t2v_loss.item(), "causal_prediction_t2v_loss", "train")
        viz.linePlot(iterId, optimizer.get_lr()[0], 'learning_rate', 'train')
        myprint(f"tboard logging took {t() - s}")

    if step % args.gradient_accumulation_steps == 0:
        # if args.fp16: Nathan
        #     # modify learning rate with special warm up BERT uses
        #     # if args.fp16 is False, BertAdam is used that handles this automatically
        #     lr_this_step = args.learning_rate * warmup_linear(
        #         global_step / num_train_optimization_steps,
        #         args.warmup_proportion,
        #     )
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr_this_step

        s = t()
        optimizer.step()
        myprint(f"Updating weights took  {t() - s}")
        s = t()
        optimizer.zero_grad()
        myprint(f"zero-ing gradients took  {t() - s}")
        # global_step += 1 #Nathan

    # Checkpointing
    global last_checkpoint_time
    if t() - last_checkpoint_time > args.checkpoint_period:
        last_checkpoint_time = t()
        checkpoint(savePath, model, optimizer, rank, device, train_dataset)


def get_epoch_file_path(savePath):
    search_regex = Path(savePath, f'start_epoch_*.json').as_posix()
    res = glob.glob(search_regex)
    assert (len(res) <= 1)
    epoch_file = res[0] if len(res) != 0 else None
    return epoch_file


def checkpoint(savePath, model, optimizer, rank, device, train_dataset):
    # Store model parameters (and load it in models on other devices)
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        print(f"\r\nStoring model and optimizer checkpoint in {savePath}")
        for obj, path in zip([model, optimizer],
                             [Path(savePath, f'{i}.checkpoint') for i in ['model', 'optimizer_state']]):
            old_ckpt_path_name = str(path) + '_old'
            if os.path.exists(path):
                os.rename(path, old_ckpt_path_name)
            torch.save(obj.state_dict(), path)
        for obj, path in zip([model, optimizer],
                             [Path(savePath, f'{i}.checkpoint') for i in ['model', 'optimizer_state']]):
            old_ckpt_path_name = str(path) + '_old'
            if os.path.exists(old_ckpt_path_name):
                os.remove(old_ckpt_path_name)
        print(f"\r\nDone storing model and optimizer checkpoint.")

    # Store where we are in the data
    train_dataset.store_checkpoint()

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    myprint("Waiting before barrier")
    dist.barrier()
    myprint("Got past barrier")

    # Store and reload over processes: model and optimizer state
    # map_location = {f'cuda:{rank_to_device(MASTER_RANK)}': str(device)}
    model.load_state_dict(
        torch.load(Path(savePath, f'model.checkpoint'), map_location=device))
    optimizer.load_state_dict(
        torch.load(Path(savePath, f'optimizer_state.checkpoint'),
                   map_location=device))



# def safely_store(checkpoint_path, object):
#     old_ckpt_path_name = str(checkpoint_path) + '_old'
#     if os.path.exists(checkpoint_path):
#         os.rename(checkpoint_path, old_ckpt_path_name)
#     torch.save(object.state_dict(), checkpoint_path)
#     if os.path.exists(old_ckpt_path_name):
#         os.remove(str(checkpoint_path) + '_old')


class TBlogger:
    def __init__(self, log_dir, exp_name):
        log_dir = Path(log_dir, exp_name)
        print(f"logging file at: {log_dir}")
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"): #TODO only do for default GPU
        self.logger.add_scalar(split + "/" + key, val, step)



if __name__ == "__main__":
    main()
