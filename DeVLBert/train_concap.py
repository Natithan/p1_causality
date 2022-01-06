# region pre-stuff

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from mycallbacks import MyProgressBar, MyModelCheckpoint, MyDataConnector

from time import time as t

first_start = t()
import glob

import os
# from memory_profiler import profile

import yaml
import sys
from pretorch_util import assign_visible_gpus
import jsonargparse


assign_visible_gpus()

# from devlbert.datasets.concept_cap_dataset import get_core_ds
from pytorch_lightning.plugins import DDPPlugin
import argparse
import json
import logging
import random
from io import open
from time import strftime
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import LearningRateMonitor
import torch.multiprocessing as mp

START = t()
import numpy as np
from tqdm import tqdm
import torch
# from tensorboardX import SummaryWriter
from pytorch_lightning.loggers.tensorboard import SummaryWriter
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from devlbert.datasets import ConceptCapLoaderTrain
import torch.distributed as dist
from constants import MODEL_CKPT_DIR, HOST
from util import MyLogger, myprint, get_rank, setup, cleanup, is_on_tpus, print_gpu_mem
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


# endregion

# region Old non-pl mains
def main():
    parser = argparse.ArgumentParser()
    parser = add_program_argparse_args(parser)

    args = parser.parse_args()

    if args.save_name is not '':
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a")
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))
    savePath = os.path.join(args.output_dir, timeStamp)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    myprint("Spawning process per GPU")
    mp.spawn(main_single_process,
             args=(args, savePath),
             nprocs=args.world_size,
             join=True)


def main_single_process(rank, args, savePath):
    device = torch.device("cuda", rank)
    myprint("Entered main_single_process")
    setup(rank=rank, world_size=args.world_size)
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
        mini=args.mini,
        shuffle=args.shuffle
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

        # Load correct epoch
        epoch_file = get_epoch_file_path(savePath)
        if epoch_file is not None:
            with open(epoch_file, 'r') as f:
                args.start_epoch = json.load(f)
            print(f"Loading start epoch from {epoch_file}")

        ckpt_load_path = Path(savePath, f"pytorch_model_{int(args.start_epoch) - 1}.bin")
        if args.continue_training and Path(ckpt_load_path).exists():
            # ckpt_load_path = os.path.join(args.from_pretrained,
            #                               "pytorch_model_{}.bin".format(int(args.start_epoch) - 1))
            # model = BertForMultiModalPreTraining.from_pretrained(ckpt_load_path, config)
            print(f"Loading model from checkpoint {ckpt_load_path}")
            # map_location = {f'cuda:{rank_to_device(MASTER_RANK)}': str(device)}

            model.module.load_state_dict(
                torch.load(ckpt_load_path, map_location=device))
            torch.cuda.empty_cache()
            # model = torch.load(ckpt_load_path, map_location=map_location)

            # if not default_gpu:
            #     print(5)
            # print("SLEEPING FOR DEBUGGING")
            # sleep(10000000)

            # # Load correct within-epoch step
            # step_file = get_step_file_path(savePath)
            # if step_file is not None:
            #     with open(step_file, 'r') as f:
            #         start_step = json.load(f)
            #     print(f"Loading start step from {step_file}")
    else:
        model = DDP(BertForMultiModalPreTraining(config).to(device), device_ids=[device])

    # Nathan: updating according to https://nvidia.github.io/apex/amp.html#transition-guide-for-old-api-users
    # if args.fp16:
    # model.half() # replacing with "model, optimizer = amp.initialize(mode, optimizer,opt_level='O2')" later on

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
        # Nathan: updating according to https://nvidia.github.io/apex/amp.html#transition-guide-for-old-api-users
        try:
            # from apex.contrib.optimizers import FP16_Optimizer
            # from apex.contrib.optimizers import FusedAdam
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False)
        # if args.loss_scale == 0:
        #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        # else:
        #     optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
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
    myprint(f'Getting to epoch loop took {t() - s} seconds')
    # random.seed(torch.distributed.get_rank() * 2000)
    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):

        ## Store epoch idx
        # delete the old file
        old_path = get_epoch_file_path(savePath)
        if old_path is not None:
            os.remove(old_path)
        # store new file
        epoch_file = Path(savePath, f'rank_{get_rank()}_start_epoch_{epochId}.json')
        with open(epoch_file, 'w') as f:
            json.dump(epochId, f)
            print(f"Stored done epochs in {epoch_file}")

        # Getting per-epoch progress bar iterator for main process
        basic_iterator = enumerate(train_dataset, 1)
        iterator = tqdm(basic_iterator,
                        total=len(train_dataset)) \
            if default_gpu else basic_iterator
        # train_dataset.
        s = t()
        first = True
        myprint(f"Loading first batch ...")
        for step, batch in iterator:
            if first:
                myprint(f"Loading first batch took {t() - s} seconds")
            else:
                myprint(f"Loading non-first batch took {t() - s} seconds")
            first = False
            s = t()
            training_step(args, batch, default_gpu, device,
                          epochId, model, optimizer, rank, savePath,
                          step, train_dataset, viz)
            myprint(f"ENTIRE TRAIN STEP TOOK {t() - s} SECONDS")
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
        # train_dataset.reset_index() Not doing this:

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

# endregion
def add_program_argparse_args(parser):
    # Required parameters
    parser.add_argument('--config', help="configuration file *.yml. Can be overriden with direct args",
                        action=jsonargparse.ActionConfigFile)

    parser.add_argument(
        "--vilbert", action="store_true",
        help="Whether to use vilbert instead of devlbert."
    )
    parser.add_argument(
        "--no_prior", action="store_true",
        help="Whether to skip weighting elements of the confounder dictionary by their prior frequency."
    )
    parser.add_argument(
        "--dependent_prior", action="store_true",
        help="Whether to weight elements of the confounder dictionary by a dependent prior, rather than "
             "an independent one."
    )
    parser.add_argument(
        "--empty_data", action="store_true",
        help="Whether to use fake empty data (to quickly check between-epoch functionality)."
    )
    parser.add_argument(
        "--train_file",
        default="data/training",
        # default="data/conceptual_caption/training",
        type=str,
        # required=True,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        # required=True,
        help="The run name as logged in wandb.",
    )
    parser.add_argument(
        "--region_mask_prob",
        default=0.15,
        type=float,
        help="Probability of masking a token during pretraining.",
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
    parser.add_argument(
        "--checkpoint_every_n_train_steps",
        default=100,
        type=int,
        help="Number of training steps (batches) between each checkpoint. To deal with within-epoch crashes.",
    )
    parser.add_argument(
        "--exact_epochs",
        default=12,
        type=int,
        help="Number of training epochs.",
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
        "--pt2_run",
        action="store_true",
        help="For the specific case of continuing pretraining with different mask rate to repro devlbert",
    )

    parser.add_argument(
        "--mystepresume",
        type=bool,
        default=True,
        help="For the specific case of continuing pretraining from a mid-epoch run based on the global step that is stored",
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
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
    # parser.add_argument(
    #     "--world_size",
    #     type=int,
    #     default=2,
    #     help="Number of processes, should equal number of GPUs you intend to use.",
    # )
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
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the concap data"
    )
    # parser.add_argument(
    #     "--gpus",
    #     nargs='+', type=int,
    #     help="which gpus to consider"
    # )
    parser.add_argument(
        "--mini", action="store_true", help="Whether to train on mini data, just to test the whole training loop"
    )
    parser.add_argument(
        "--dummy_model", action="store_true", help="Whether to use a mini, dummy model that doesn't "
                                                   "take a lot of GPU space"
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
    parser.add_argument(
        "--visible_gpus",
        default=None,
        type=str,
        help="If set, overrides the default which shows GPUs on which no other people are running "
             "(for setting when sharing GPUs with other people)",
    )
    return parser


def add_jsonargparse_args(cls, parent_parser: jsonargparse.ArgumentParser) -> jsonargparse.ArgumentParser:
    r"""Extends existing argparse by default `Trainer` attributes.

    Args:
        cls: Lightning class
        parent_parser:
            The custom cli arguments parser, which will be extended by
            the Trainer default arguments.

    Only arguments of the allowed types (str, float, int, bool) will
    extend the `parent_parser`.

    Examples:
        >>> import argparse
        >>> from pytorch_lightning import Trainer
        >>> parser = argparse.ArgumentParser()
        >>> parser = Trainer.add_argparse_args(parser)
        >>> args = parser.parse_args([])
    """
    # parser = jsonargparse.ArgumentParser(
    #     parents=[parent_parser],
    #     add_help=False
    # )
    parser = parent_parser

    blacklist = ['kwargs']
    depr_arg_names = cls.get_deprecated_arg_names() + blacklist

    allowed_types = (str, int, float, bool)
    from pytorch_lightning.utilities.argparse import parse_args_from_docstring, get_init_arguments_and_types, \
        _gpus_allowed_type, _int_or_float_type
    from pytorch_lightning.utilities import parsing
    args_help = parse_args_from_docstring(cls.__init__.__doc__ or cls.__doc__)
    for arg, arg_types, arg_default in (at for at in get_init_arguments_and_types(cls) if at[0] not in depr_arg_names):
        arg_types = [at for at in allowed_types if at in arg_types]
        if not arg_types:
            # skip argument with not supported type
            continue
        arg_kwargs = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type = parsing.str_to_bool
            elif str in arg_types:
                use_type = parsing.str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            use_type = arg_types[0]

        if arg == 'gpus' or arg == 'tpu_cores':
            use_type = _gpus_allowed_type

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if arg == 'track_grad_norm':
            use_type = float

        parser.add_argument(
            f'--{arg}',
            dest=arg,
            default=arg_default,
            type=use_type,
            help=args_help.get(arg),
            **arg_kwargs,
        )

    return parser


# @profile(stream=PROFILING_LOG_FILE_HANDLE)
# @profile
def main_pl():
    s = t()
    parser = jsonargparse.ArgumentParser()
    parser = add_program_argparse_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    # parser = Trainer.add_argparse_args(parser)
    # parser = add_jsonargparse_args(Trainer, parser)

    parser.add_class_arguments(Trainer, 'trainer', as_group=True)
    args = parser.parse_args()

    if args.run_name is not None:
        wandb_logger = WandbLogger(project="sceps", name=args.run_name)
    else:
        wandb_logger = WandbLogger(project="sceps")
    if not args.vilbert:
        from devlbert.devlbert import BertForMultiModalPreTraining, BertConfig
    else:
        from devlbert.vilbert import BertForMultiModalPreTraining, BertConfig
    if args.trainer.min_epochs != args.trainer.max_epochs:
        raise NotImplementedError
    else:
        args.num_train_epochs = args.trainer.min_epochs
    if not is_on_tpus():
        args.trainer.tpu_cores = None
    else:
        if HOST == 'LIIR':
            if args.trainer.gpus == None:
                args.trainer.gpus = 4
        elif HOST == 'VSC':
            if args.trainer.gpus == None:
                args.trainer.gpus = 8
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.baseline:
        raise NotImplementedError
    if args.exact_epochs > 0:
        args.trainer.min_epochs = args.trainer.max_epochs = args.exact_epochs
        myprint(f"Training for exactly {args.exact_epochs} epochs")

    logger.info(yaml.dump(args))

    # region Getting BertConfig and tweaking it based on args
    config = BertConfig.from_json_file(args.config_file)
    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]
    if args.without_coattention:
        config.with_coattention = False
    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False
    # endregion

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # seed_everything(args.seed DONE IN PL ALREADY

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.from_pretrained != 'false':
        model = BertForMultiModalPreTraining.from_pretrained(args.from_pretrained, config, args=args)
    else:
        model = BertForMultiModalPreTraining(config)

    if args.pt2_run:
        PT2_REGION_MASK_PROB = 0.3
        assert args.region_mask_prob == PT2_REGION_MASK_PROB
        list_of_pt2_ckpts = glob.glob(f'{args.output_dir}/epoch=*{PT2_REGION_MASK_PROB}.ckpt')
        if len(list_of_pt2_ckpts) > 0:
            myprint("args.pt2 true, but pt2 intermediate checkpoints found. Restarting from those.")
            latest_ckpt = max(list_of_pt2_ckpts, key=os.path.getctime)
            args.trainer.resume_from_checkpoint = latest_ckpt
        else:
            PT1_REGION_MASK_PROB = 0.15
            list_of_pt1_ckpts = glob.glob(f'{args.output_dir}/epoch=*{PT1_REGION_MASK_PROB}.ckpt')
            print(list_of_pt1_ckpts)
            myprint(f"Loading pt1 checkpoint from {f'{args.output_dir}/epoch=*{PT1_REGION_MASK_PROB}.ckpt'}")
            latest_pt1_ckpt = max(list_of_pt1_ckpts, key=os.path.getctime)
            model = BertForMultiModalPreTraining.load_from_checkpoint(latest_pt1_ckpt, config=config, args=args)
    else:
        if args.mystepresume:
            list_of_ckpt = glob.glob(
                f'{args.output_dir}/epoch=*.ckpt')  # * means all if need specific format then *.csv
            if len(list_of_ckpt) != 0:
                # print(list_of_ckpt)
                latest_ckpt = max(list_of_ckpt, key=os.path.getctime)
                # print(latest_ckpt)
                args.trainer.resume_from_checkpoint = latest_ckpt

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_filename = f'{{epoch}}-{{step}}-{args.region_mask_prob}'
    checkpoint_callback = MyModelCheckpoint(dirpath=args.output_dir, save_top_k=-1,
                                            every_n_train_steps=args.checkpoint_every_n_train_steps,
                                            filename=checkpoint_filename)
    progressbar_callback = MyProgressBar()
    trainer = pl.Trainer.from_argparse_args(args.trainer,  # auto_scale_batch_size='binsearch',
                                            plugins=[DDPPlugin(
                                                find_unused_parameters=False)] if args.trainer.tpu_cores is None else [],
                                            # from https://pytorch-lightning.readthedocs.io/en/latest/benchmarking/performance.html#when-using-ddp-set-find-unused-parameters-false
                                            callbacks=[lr_monitor, checkpoint_callback, progressbar_callback],
                                            profiler='simple',
                                            logger=wandb_logger)

    # Overriding DataConnector to start batch_idx at nonzero position after resuming mid-epoch checkpoint
    trainer.data_connector = MyDataConnector(trainer, exp_args=args)
    # init data flags
    trainer.data_connector.on_trainer_init(
        args.trainer.check_val_every_n_epoch, args.trainer.reload_dataloaders_every_epoch,
        args.trainer.prepare_data_per_node
    )
    print_gpu_mem()
    myprint("***** Running training *****")
    myprint(f'Getting to trainer.fit took {t() - first_start} seconds')
    # trainer.tune(model,train_dataloader=train_dataset)
    # trainer.fit(model, train_dataloader=train_dataset)

    if args.dummy_model:
        print("+"*100,"USING DUMMY MODEL","+"*100)
        model = DummyModel(args)
        trainer.callbacks = []
        trainer.logger = None

    trainer.fit(model)


class DummyModel(pl.LightningModule):
    """BERT model with multi modal pre-training heads.
    """

    def __init__(self,args):
        super().__init__()
        self.linear = torch.nn.Linear(10,1)
        self.automatic_optimization = True
        self.args = args

    def forward(self,image_feat):
        pred = self.linear(image_feat[:,0,:10]).squeeze()
        target = image_feat[:,0,11]
        loss = torch.abs(pred-target).sum()
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, \
        image_ids, causal_label_t, causal_label_v = (
            batch
        )
        loss = self(image_feat)
        return {'loss': loss}

    def train_dataloader(self):
        if (not hasattr(self,'train_dataset')) or self.train_dataset is None:
            tokenizer = BertTokenizer.from_pretrained(
                self.args.bert_model, do_lower_case=True
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def training_step(args, batch, default_gpu, device, epochId, model, optimizer, rank, savePath, step, train_dataset,
                  viz):
    batch = tuple(tup.cuda(device=device, non_blocking=True) for tup in batch)
    input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, \
    image_ids, causal_label_t, causal_label_v = (
        batch
    )
    s = t()
    myprint(f'Doing Model input --> model output losses')
    # test comment VSC
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
        # Nathan: updating according to https://nvidia.github.io/apex/amp.html#transition-guide-for-old-api-users
        # optimizer.backward(loss)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    myprint(f'Calculating gradients took {t() - s}')
    # if math.isnan(loss.detach().item()):
    #     myprint("math.isnan(loss.item())")
    #     pdb.set_trace()
    if default_gpu:
        s = t()
        myprint(f"tboard logging ...")
        iterId = (step) + (epochId * len(train_dataset))
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
        # Nathan: not trying to imitate BertAdam, but just using FusedAdam, based on this discussion https://github.com/huggingface/transformers/issues/420
        # if args.fp16:
        #     # modify learning rate with special warm up BERT uses
        #     # if args.fp16 is False, BertAdam is used that handles this automatically
        #     lr_this_step = args.learning_rate * warmup_linear(
        #         global_step / num_train_optimization_steps,
        #         args.warmup_proportion,
        #     )
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr_this_step

        myprint(f"Updating weights ...")
        s = t()
        optimizer.step()
        myprint(f"Updating weights took  {t() - s}")
        myprint(f"Zeroing gradients ...")
        s = t()
        optimizer.zero_grad()
        myprint(f"zero-ing gradients took  {t() - s}")
        # global_step += 1 #Nathan

    # Checkpointing
    global last_checkpoint_time
    if t() - last_checkpoint_time > args.checkpoint_period:
        last_checkpoint_time = t()
        checkpoint(savePath, model, optimizer, rank, device)


def get_step_file_path(savePath):
    search_regex = Path(savePath, f'rank_{get_rank()}_start_step_*.json').as_posix()
    res = glob.glob(search_regex)
    assert (len(res) <= 1)
    step_file = res[0] if len(res) != 0 else None
    return step_file


def get_epoch_file_path(savePath):
    search_regex = Path(savePath, f'rank_{get_rank()}_start_epoch_*.json').as_posix()
    res = glob.glob(search_regex)
    assert (len(res) <= 1)
    epoch_file = res[0] if len(res) != 0 else None
    return epoch_file


def checkpoint(savePath, model, optimizer, rank, device):
    # Store model parameters (and load it in models on other devices)
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        myprint(f"\r\nStoring model and optimizer checkpoint in {savePath}")
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
        myprint(f"\r\nDone storing model and optimizer checkpoint.")

    # # train_dataset.store_checkpoint() ONLY STORING HOW MUCH DONE, NOT WHAT DONE
    # # # Store where we are in the data
    # # delete the old file
    # old_path = get_step_file_path(savePath)
    # if old_path is not None:
    #     os.remove(old_path)
    # # store new file
    # step_file = Path(savePath, f'rank_{get_rank()}_start_step_{step}.json')
    # with open(step_file, 'w') as f:
    #     json.dump(step, f)
    #     print(f"Stored done epochs in {step_file}")

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

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)


if __name__ == "__main__":
    # main()
    main_pl()
