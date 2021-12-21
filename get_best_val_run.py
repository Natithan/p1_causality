import os

import re
from collections import OrderedDict

import glob
import argparse

from constants import PROJECT_ROOT_DIR, MINI_FT_EPOCHS, BIGSTORAGE_ROOT_DIR, VSC_BIGSTORAGE_ROOT_DIR

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
    # 'gimli_1': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/finetune_ir.o50775714',
    # 'gimli_2': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/finetune_ir.o50775710',
    # 'v4': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/finetune_ir.o50775716',
    # 'v5': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_v5.o50778908',
    # 'v6': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/finetune_ir.o50775712',
    'v6': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_v6.o50962659',
    # 'literal_copy': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_literal_copy.o50778116',
    # 'no_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_no_prior.o50779422',
    # 'vilbert': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_vilbert.o50778570',
    # 'dependent_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_dependent_prior.o50783046',
    'dependent_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_dependent_prior.o50964074',
    'dv7': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/ir_dv7.o50963490',
}
vqa_log_files = {
    # 'gimli_1': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_gimli_1.o50775713',
    # 'gimli_2': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_gimli_2.o50775709',
    # 'v4': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_v4.o50775715',
    # 'v5': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_v5.o50778907',
    # 'v6': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_v6.o50779421',
    # 'literal_copy': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_literal_copy.o50778382',
    # 'no_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_no_prior.o50778507',
    # 'vilbert': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_vilbert.o50775717',
    # 'dependent_prior': f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/vqa_dependent_prior.o50781547',
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
    parser.add_argument(
        "--mini", action="store_true", help="Whether working with a mini-debugging setup"
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
        print_path_for_run_name(args.run_name, args.metric, args.tmp_file, args.mini)

def get_log_file_path(run_name,metric,mini=False):
    if metric == 'ir':
        log_files = ir_log_files
    elif metric == 'vqa':
        log_files = vqa_log_files
    else:
        raise ValueError(f"Metric {metric} not supported")
    if run_name in log_files:
        p = log_files[run_name]
    else:
        mini_prefix = "" if not mini else "mini_"
        path_search_string = f'{PROJECT_ROOT_DIR}/vsc_batch_terminal/after_pretrain/{mini_prefix}{metric}_{run_name}.o*'
        paths = glob.glob(path_search_string)
        print(f"Searching for latest log file matching {path_search_string}")
        latest_file = max(paths, key=os.path.getctime)
        p = latest_file
        print(f"Found {p} as matching file")
    return p

def distort_log_line(line):
    i1 = line.find("Saving fine - tuned")
    i2 = line.find("Validation [") + 10
    if i1 != -1:
        return line[:i1] + "CKPT FINDER FOUND LINE: " + line[i1:]
    elif i2 != -1:
        return line[:i2] + " CKPT FINDER FOUND LINE:" + line[i2:]
    else:
        print("Not distorting line")
        return line

def print_path_for_run_name(run_name, metric, tmp_file=None,mini=False):
    # p = log_files[run_name]
    p = get_log_file_path(run_name,metric,mini)
    ckpt_path_lines = []
    val_score_lines = []
    world_size_found = False
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if (not world_size_found) and ("\'--world_size\'" in line):
                ws_match_regex = '.* \'--world_size\', \'([0-9]+)'
                world_size = int(re.match(ws_match_regex, line).groups()[0])
            if '** ** * Saving fine - tuned model on ' in line:
                ckpt_path_lines.append(line)
            if "Validation [" in line:
                line_without_timestamp = line[line.index("Validation"):]
                val_score_lines.append(line_without_timestamp)
    nb_epochs = 20 if not mini else MINI_FT_EPOCHS
    print("ckpt_path_lines\r\n",*[distort_log_line(line) for line in ckpt_path_lines])
    print("val_score_lines\r\n",*[distort_log_line(line) for line in val_score_lines])
    assert len(ckpt_path_lines)*world_size == len(val_score_lines)
    assert len(ckpt_path_lines) == nb_epochs
    ckpts_and_scores = []
    val_score_line_groups = [val_score_lines[i*world_size:i*world_size + world_size] for i in range(nb_epochs)]
    for ckpt_line, score_line_group in zip(ckpt_path_lines,val_score_line_groups):
        # print(ckpt_line)
        # print(score_line)
        ckpt_match_regex = '.*Saving fine - tuned model on .* in (.+) \*\* \*\* \*.*'
        ckpt_path = re.match(ckpt_match_regex, ckpt_line).groups()[0]
        # print("ckpt_path",ckpt_path)
        ema_ckpt_path = ckpt_path[:-4] + "_ema.bin"
        score_match_regex = '.* score (.+) \\n.*'
        # scores = [float(re.match(score_match_regex, l).groups()[0]) for l in lines[5 * i + 1:5 * i + 5]]
        # score = sum(scores) / len(scores)
        scores = [float(re.match(score_match_regex, l).groups()[0]) for l in score_line_group]
        score = sum(scores) / len(scores)
        ckpts_and_scores.append((ema_ckpt_path, score))
    print(max(ckpts_and_scores, key=lambda item: item[1])[0])
    if tmp_file is not None:
        best_ckpt_and_score = max(ckpts_and_scores, key=lambda item: item[1])
        best_ckpt_path = best_ckpt_and_score[0]
        if (VSC_BIGSTORAGE_ROOT_DIR in best_ckpt_path) and (VSC_BIGSTORAGE_ROOT_DIR != BIGSTORAGE_ROOT_DIR):
            best_ckpt_path = best_ckpt_path.replace(VSC_BIGSTORAGE_ROOT_DIR, BIGSTORAGE_ROOT_DIR)
            print(f"Adjusting best checkpoint path to non-VSC host, now: {best_ckpt_path}")
        # print(f"Using max top of checkpoints/scores {ckpts_and_scores}")
        print(best_ckpt_path,file=open(tmp_file,'w'))


if __name__ == '__main__':
    main()