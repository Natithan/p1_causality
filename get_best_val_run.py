import os

import re
import glob
import argparse

from constants import PROJECT_ROOT_DIR
ALL_RUN_NAMES = [
        'gimli_1',
        'gimli_2',
        'v4',
        'v5',
        'v6',
        'literal_copy',
        'no_prior',
        'vilbert']# , 'dependent_prior'


ir_log_files = {
    'gimli_1': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/finetune_ir.o50775714',
    'gimli_2': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/finetune_ir.o50775710',
    'v4': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/finetune_ir.o50775716',
    'v5': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_v5.o50778908',
    'v6': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/finetune_ir.o50775712',
    'literal_copy': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_literal_copy.o50778116',
    'no_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_no_prior.o50779422',
    'vilbert': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_vilbert.o50778570',
    'dependent_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_dependent_prior.o50783046',
}
vqa_log_files = {
    'gimli_1': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_gimli_1.o50775713',
    'gimli_2': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_gimli_2.o50775709',
    'v4': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_v4.o50775715',
    'v5': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_v5.o50778907',
    'v6': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_v6.o50779421',
    'literal_copy': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_literal_copy.o50778382',
    'no_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_no_prior.o50778507',
    'vilbert': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_vilbert.o50775717',
    'dependent_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_dependent_prior.o50781547',
}
def add_program_argparse_args(parser):
    parser.add_argument(
        "--run_name",
        type=str,
        # required=True,
        help="The run for which to find the best validation-scored checkpoint",
    )
    parser.add_argument(
        "--tmp_file",
        type=str,
        required=True,
        help="File where to return the answer",
    )
    parser.add_argument(
        "--metric",
        type=str,
        # required=True,
        help="vqa or ir",
    )
    return parser

def main():
    parser = argparse.ArgumentParser()
    parser = add_program_argparse_args(parser)
    args = parser.parse_args()
    if args.run_name == 'all':
        for r in ALL_RUN_NAMES:
            for m in ['ir','vqa']:
                print_path_for_run_name(r, m)
    else:
        # log_files = ir_log_files if args.metric == 'ir' else vqa_log_files
        print_path_for_run_name(args.run_name, args.metric, args.tmp_file)

def get_log_file_path(run_name,metric):
    if metric == 'ir':
        log_files = ir_log_files
    elif metric == 'vqa':
        log_files = vqa_log_files
    else:
        raise ValueError(f"Metric {metric} not supported")
    if run_name in log_files:
        p = log_files[run_name]
    else:
        paths = glob.glob(f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/{metric}_{run_name}.o*')
        latest_file = max(paths, key=os.path.getctime)
        p = latest_file
    return p

def print_path_for_run_name(run_name, metric, tmp_file=None):
    # p = log_files[run_name]
    p = get_log_file_path(run_name,metric)
    lines = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if '** ** * Saving fine - tuned model on ' in line:
                lines.append(line)
            if "Validation [" in line:
                lines.append(line)
    ckpts_and_scores = []
    for i in range(20):
        ckpt_path = re.match('.*Saving fine - tuned model on .* in (.+) \*\* \*\* \*.*', lines[5 * i]).groups()[0]
        ema_ckpt_path = ckpt_path[:-4] + "_ema.bin"
        scores = [float(re.match('.* score (.+) \\n.*', l).groups()[0]) for l in lines[5 * i + 1:5 * i + 5]]
        score = sum(scores) / len(scores)
        ckpts_and_scores.append((ema_ckpt_path, score))
    print(max(ckpts_and_scores, key=lambda item: item[1])[0])
    if tmp_file is not None:
        print(max(ckpts_and_scores, key=lambda item: item[1])[0],file=open(tmp_file,'w'))


if __name__ == '__main__':
    main()