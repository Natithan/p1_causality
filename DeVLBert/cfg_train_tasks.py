import os

from absl import flags
import sys

FGS = flags.FLAGS
flags.DEFINE_string("bert_model","bert-base-uncased","Bert pre-trained model selected in the list: bert-base-uncased, "
         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
flags.DEFINE_string("from_pretrained","bert-base-uncased","Bert pre-trained model selected in the list: bert-base-uncased, "
         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
flags.DEFINE_string("output_dir","/cw/working-gimli/nathan/ckpts_downstream","The output directory where the model checkpoints will be written.")
flags.DEFINE_string("config_file","config/bert_config.json","The config file which specified the model details.")
flags.DEFINE_string("save_name","","save name for training.")
flags.DEFINE_string("optimizer","BertAdam","whether use chunck for parallel training.")
flags.DEFINE_string("tasks","","1-2-3... training task separate by -")
flags.DEFINE_string("lr_scheduler","mannul","whether use learning rate scheduler.")
flags.DEFINE_string("mode","","For compatibility with PyCharm python console")

flags.DEFINE_bool("mini", False, "")
flags.DEFINE_bool("no_cuda", False, "Whether not to use CUDA when available")
flags.DEFINE_bool("fp16", False, "Whether to use 16-bit float precision instead of 32-bit")
flags.DEFINE_bool("vision_scratch", False, "whether pre-trained the image or not.")
flags.DEFINE_bool("baseline", False, "whether use single stream baseline.")
flags.DEFINE_bool("compact", False, "whether use compact vilbert model.")
flags.DEFINE_bool("use_ema", False, "whether to use EMA.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. True for uncased models, False for cased models.")
flags.DEFINE_bool("in_memory", False, "whether use chunck for parallel training.")
flags.DEFINE_bool("vilbert", False, "whether to use vilbert instead of devlbert")

flags.DEFINE_integer("evaluation_interval", 1, "evaluate very n epoch.")
flags.DEFINE_integer("world_size", len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")), "Number of processes, should equal number of GPUs you intend to use.")
flags.DEFINE_integer("num_train_epochs", 20, "Total number of training epochs to perform.")
flags.DEFINE_integer("local_rank", -1, "local_rank for distributed training on gpus")
flags.DEFINE_integer("seed", 0, "random seed for initialization")
flags.DEFINE_integer("gradient_accumulation_steps", 1, "Number of updates steps to accumualte before performing a backward")
flags.DEFINE_integer("num_workers", 16, "Number of workers in the dataloader.")
flags.DEFINE_integer("freeze", -1, "till which layer of textual stream of vilbert need to fixed.")
flags.DEFINE_integer("batch_size", 256, "till which layer of textual stream of vilbert need to fixed.")

flags.DEFINE_float("learning_rate",2e-5,"The initial learning rate for Adam.")
flags.DEFINE_float("warmup_proportion",0.1,"Proportion of training to perform linear learning rate warmup for. "
         "E.g., 0.1 = 10%% of training.")
flags.DEFINE_float("ema_decay_ratio",0.9999,"EMA dacay ratio.")
flags.DEFINE_float("loss_scale",0,"Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
         "0 (default value): dynamic loss scaling.\n"
         "Positive power of 2: static loss scaling value.\n")
flags.DEFINE_float("use_chunk",0,"whether use chunck for parallel training.")
flags.DEFINE_float("mini_fraction",1/120,"Fraction of data to consider when doing mini run")

# eval_retrieval.py flags

flags.DEFINE_string("split","","which split to use.")

flags.DEFINE_bool("zero_shot", False, "")



FGS(sys.argv)