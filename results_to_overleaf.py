import os, glob, io
import seaborn as sns
import re
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import colors
from constants import PROJECT_ROOT_DIR
from pathlib import Path
import numpy as np

PP_OUTPUT_DIR = f'{PROJECT_ROOT_DIR}/PP_output'
dev_repro_runs = ('gimli_1', 'gimli_2', 'v4', 'v5', 'v6')
dev_96_repro_runs = ('gimli_1', 'gimli_2')
dev_128_repro_runs = ('v4', 'v5', 'v6')
vi_repro_runs = ('vilbert',) + tuple([f'vilbert_{i}' for i in range(2, 6)])
literal_run = 'literal_copy'
ALL_RUNS = dev_repro_runs + vi_repro_runs + ('literal_copy',) + (
    'no_prior', 'dependent_prior', 'devlbert_reported', 'vilbert_reported', 'apr_gimli_1',
    'apr_10eps_gimli_1')
# versions = ['best_val','default']
versions = ['best_val']
# versions = ['default']
ALL_SCORES = ['mAP', 'zsir'] + [f'{m}_{v}' for m in ['ir', 'vqa'] for v in versions]
ALL_RESULTS = ALL_SCORES + ['avgAtt']
VQA_RESULTS = pd.read_csv('vqa_evalai_results.csv')
DS_COLUMNS = [f'ir_{v} R@{num}' for v in versions for num in [1, 5, 10]] + [f'zsir R@{num}' for num in [1, 5, 10]] + [
    f'vqa_{v} test-dev' for v in versions]
mAP_COLUMNS = ['mAP_devlbert', 'excess_mAP']
NO_PRIOR_LTX_NAME = '\\noPrior'
DEP_PRIOR_LTX_NAME = '\\depPrior'
LITERAL_COPY_LTX_NAME = '\\ckptCopy'
DEVLBERT_LTX_NAME = '\\D'
NB_CAUSES = 4
VILBERT_LTX_NAME = '\\VI'
BEST_REPRO_RUN = 'v4'
# DS_COLUMNS = ['ir_best_val R@1', 'ir_best_val R@5', 'ir_best_val R@10', 'zsir R@1', 'zsir R@5', 'zsir R@10',
#                                               'vqa_best_val test-dev']
reduced_version = [versions[0]]
REPORTED_COL_NAMES = [f'ir_{v} R@{num}' for v in reduced_version for num in [1, 5, 10]] + [f'zsir R@{num}' for num in
                                                                                           [1, 5, 10]] + [
                         'vqa_best_val test-std'] + [f'vqa_{v} test-dev' for v in reduced_version]

simple_summary_names = ["D", "V"]
DS_row_rename_dict = {
    'average_repro': f'{DEVLBERT_LTX_NAME} repro (5 run avg)',
    f'summary_repro {simple_summary_names[0]}': f'{DEVLBERT_LTX_NAME} repro (5 run avg ± stdev)',
    f'summary_repro {simple_summary_names[1]}': f'{VILBERT_LTX_NAME} repro (5 run avg ± stdev)',
    'literal_copy': LITERAL_COPY_LTX_NAME,
    'devlbert_reported': f'{DEVLBERT_LTX_NAME} reported',
    'vilbert': f'{VILBERT_LTX_NAME} repro',
    'vilbert_reported': f'{VILBERT_LTX_NAME} reported',
    'dependent_prior': DEP_PRIOR_LTX_NAME,
    'no_prior': NO_PRIOR_LTX_NAME,
    BEST_REPRO_RUN: f'{DEVLBERT_LTX_NAME} repro (best run)'
}

GREEN_CM = sns.light_palette("green", as_cmap=True)
GREY_CM = sns.light_palette("grey", as_cmap=True)
USED_CM = GREY_CM
DS_latex_rows = ['devlbert_reported', 'vilbert_reported'] + [f'summary_repro {name}' for name in
                                                             simple_summary_names] + ['literal_copy',
                                                                                      'dependent_prior',
                                                                                      'no_prior']
mAP_latex_rows = [f'summary_repro {simple_summary_names[0]}', 'literal_copy',
                  'dependent_prior',
                  'no_prior']


def get_path(run_name, type):
    d = {
        'mAP': f'{PP_OUTPUT_DIR}/{run_name}/mAP_output/avg_mAP_comparison_*_90760*.csv',
        'zsir': f'{PP_OUTPUT_DIR}/{run_name}/ZSIR/default/test_r_scores.json',
        'ir_default': f'{PP_OUTPUT_DIR}/{run_name}/IR/default/test_r_scores.json',
        'ir_best_val': f'{PP_OUTPUT_DIR}/{run_name}/IR/best_val/test_r_scores.json',
        'vqa_default': f'{PP_OUTPUT_DIR}/{run_name}/VQA/default/eval.txt',
        'vqa_best_val': f'{PP_OUTPUT_DIR}/{run_name}/VQA/best_val/eval.txt',
        'avgAtt': f'{PP_OUTPUT_DIR}/{run_name}/avgAtt_output/avgAtt_*_90760*.csv',
    }

    files = glob.glob(d[type])
    assert files, f"No match found for {d[type]}"
    return files[0]


def get_scores_for_run_name(r, type='mAP'):
    files = glob.glob(f'{PROJECT_ROOT_DIR}/PP_output/{r}/mAP_output/avg_mAP_comparison_*_90760*.csv')
    assert files, f"No match found for {PROJECT_ROOT_DIR}/PP_output/{r}/mAP_output/avg_mAP_comparison_*_90760*.csv"

    return pd.read_csv(files[0])


def path2df(run, result):
    path = get_path(run, result)
    if result == 'mAP':
        df = pd.read_csv(path)
    elif result in ('zsir', 'ir_best_val', 'ir_default'):
        df = pd.read_json(path).transpose()
        df.columns = [f"{result} R@1", f"{result} R@5", f"{result} R@10", f"{result} Median Rank",
                      f"{result} Mean Rank"]
    elif result in ('vqa_best_val', 'vqa_default'):
        # with open("/cw/liir/NoCsBack/testliir/nathan/p1_causality/PP_output/gimli_1/VQA/best_val/eval.txt", 'r') as f:
        #     text = f.readlines()[0]
        # df = pd.DataFrame([re.search('loss (.+?) score (.+?)', text).groups(1)], columns=['VQA Val Loss', 'VQA Score'])
        if 'best_val' in result:
            name = f'{run} best_val'
        else:
            name = run
        vqa_row = VQA_RESULTS.loc[VQA_RESULTS['Method Name'] == name]
        # df = pd.DataFrame(pd.read_json(vqa_row['Result File'].iloc[0], typ='series')[0]).transpose().rename(
        #     columns={'overall': f'{result} test-dev'}, index={'test-dev': 0})
        df = pd.DataFrame(pd.read_json(vqa_row['Result File'].iloc[0], typ='series')[0]).transpose().rename(
            index={'test-dev': 0})
        df = df.rename(columns={'overall': 'test-dev'})
        df = df.rename(columns=lambda col_name: f'{result} {col_name}')
        # df = df.reset_index().rename(columns={'index': 'Run name'})
    elif result == 'avgAtt':
        full_df = pd.read_csv(path)
        sorted_df = full_df.set_index('Unnamed: 0').sort_values(by='Counts', ascending=False)
        a = {sorted_df.index[i]: sorted_df.iloc[i].iloc[1:].sort_values(ascending=False).iloc[:NB_CAUSES] for i in
             range(1,11)} # Skipping 'background'
        df = pd.DataFrame(
            {effect: [f'{cls}: {str(round(score, 3))}' for cls, score in zip(causes.index, causes)] for effect, causes
             in a.items()}).transpose()
        c_df = pd.DataFrame(
            {effect: [score for score in causes] for effect, causes
             in a.items()}).transpose()
        df.name = run
        c_df.name = run
        return (df, c_df)
    else:
        raise ValueError(f'{result} not a valid option')

    return df


def shoulddrop(col):
    return any([substr in col for substr in
                ['mAP_baseline', 'mAP_baseline_emp', 'batch_num', 'zsir Median Rank', 'zsir Mean Rank',
                 'ir Median Rank', 'ir Mean Rank', 'other', 'number', 'yes/no']])


def rename_col_for_latex(col):
    new_name = ''
    if col.startswith('ir'):
        new_name += 'IR '
    elif col.startswith('vqa'):
        new_name += 'VQA test-dev'
    elif col.startswith('zsir'):
        new_name += 'ZSIR '
    elif col.startswith('mAP_devlbert'):
        new_name += 'mAP score'
    elif col.startswith('excess_mAP'):
        new_name += 'mAP excess over random baseline'
    re_match = re.match('.*R@([0-9]+)', col)
    if bool(re_match):
        num = re_match.groups()[0]
        new_name += f'R@{num}'

    if '_best_val' in col:
        # new_name += ' (B)'
        new_name += ''
    elif '_default' in col:
        new_name += ' (L)'
    return new_name


def b_g(target, source, cmap='PuBu', low=0, high=0):
    isCol = len(target.shape) < 2
    if isCol:
        a = source.loc[:, target.name].copy()
        mx = a.max()
        mn = a.min()
    else:
        a = source.copy()
        mx = a.max().max()
        mn = a.min().min()
    # # Pass the columns from Dataframe A
    # print(source, target, target.name)
    rng = mx - mn
    norm = colors.Normalize(mn - (rng * low),
                            mx + (rng * high))
    normed = norm(a.values)
    COLOR_THRESHOLD = 0.4 if USED_CM == GREEN_CM else 0.6
    if isCol:
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        tc = ['white' if sum(colors.hex2color(i)) / 3 < COLOR_THRESHOLD else 'black' for i in c]
        return [f'background-color: {color}; color:{textcolor}' for color, textcolor in zip(c, tc)]
    else:
        c = [[colors.rgb2hex(x) for x in row] for row in plt.cm.get_cmap(cmap)(normed)]
        tc = [['white' if sum(colors.hex2color(i)) / 3 < COLOR_THRESHOLD else 'black' for i in r] for r in c]
        return np.array(
            [[f'background-color: {color}; color:{textcolor}' for color, textcolor in zip(row, trow)] for row, trow in
             zip(c, tc)])
    # TODO fix light text here too
    # tc = [colors.rgb2hex(0) for _ in normed]
    # return [f'background-color: {col}, color: {textcol}' for col,textcol in zip(c,tc)]


def main():
    mAP_scores = []
    baseline_scores = []
    frames = []
    avgAtt_dfs = []
    c_avgAtt_dfs = []
    for run in ALL_RUNS:
        columns = [pd.DataFrame([[run]], columns=['Run name'])]
        if run == 'devlbert_reported':
            columns += [pd.DataFrame([[61.6, 87.1, 92.6, 36.0, 67.1, 78.3, 71.1, 71.5]],
                                     columns=REPORTED_COL_NAMES)]

        elif run == 'vilbert_reported':
            columns += [pd.DataFrame([[58.2, 84.9, 91.5, 31.9, 61.1, 72.8, 70.6, 70.9]],
                                     columns=REPORTED_COL_NAMES)]
        else:
            for result in ALL_RESULTS:
                # if run == 'dependent_prior' and result in ('ir_best_val', 'ir_default'):
                #     print(f"SKIPPING {run} - {result} FOR NOW AS NOT READY YET")
                #     continue
                if run == 'vilbert':
                    if result == 'mAP':
                        print(f"SKIPPING {run, result} FOR NOW AS NOT READY YET")
                        columns.append(pd.DataFrame([["/", "/", "/", "/", "/"]],
                                                    columns=['mAP_devlbert', 'mAP_baseline', 'mAP_baseline_emp',
                                                             'batch_num', 'excess_mAP']))
                        continue
                    elif result == 'avgAtt':
                        print(f"SKIPPING {run, result} FOR NOW AS NOT READY YET")
                        continue
                if run in ('apr_gimli_1', 'apr_10eps_gimli_1') and result != 'ir_best_val':
                    print(f"SKIPPING {run, result} FOR NOW AS NOT READY YET")
                    continue
                print(run, result)
                res = path2df(run, result)
                if not result == 'avgAtt':
                    df = res
                    columns.append(df)
                else:
                    df, c_df = res
                    avgAtt_dfs.append(df)
                    c_avgAtt_dfs.append(c_df)

        new_frame = pd.concat(columns, axis=1)
        frames.append(new_frame)
    total_df = pd.concat(frames)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print_avgAtt(avgAtt_dfs, c_avgAtt_dfs)
    # print(multi_for_latex.to_latex(header=False, bold_rows=True,escape=False,column_format=f'p{{3cm}}l{"l"*NB_CAUSES}'))

    for runs, name in zip([dev_repro_runs, vi_repro_runs, dev_96_repro_runs, dev_128_repro_runs], simple_summary_names + ["D96", "D128"]):
        relevant_slice = total_df.loc[total_df['Run name'].isin(runs)]
        average_repro_df = pd.concat([pd.DataFrame([[f"average_repro {name}"]], columns=['Run name']), pd.DataFrame(
            relevant_slice.mean()).transpose()], axis=1)
        stdev_repro_df = pd.concat([pd.DataFrame([[f"stdev_repro {name}"]], columns=['Run name']), pd.DataFrame(
            relevant_slice.std()).transpose()], axis=1)
        total_df = pd.concat([total_df, average_repro_df, stdev_repro_df])

    print([c for c in total_df.columns if shoulddrop(c)])
    reduced_total_df = total_df.drop([c for c in total_df.columns if shoulddrop(c)], axis=1)
    df = reduced_total_df
    df.set_index('Run name', inplace=True)
    # DS_col_rename_dict = {'zsir R@1': 'ZSIR R@1', 'zsir R@5': 'ZSIR R@5', 'zsir R@10': 'ZSIR R@10', 'ir R@1': 'IR R@1',
    #                       'ir R@5': 'IR R@5', 'ir R@10': 'IR R@10'}
    DS_latex_df = df

    color_source_DS_df = df.loc[
        [f'average_repro {e[len("summary_repro "):]}' if 'summary_repro' in e else e for e in
         DS_latex_rows], DS_COLUMNS].rename(
        columns=rename_col_for_latex, index=DS_row_rename_dict)

    color_source_mAP_df = df.loc[
        [f'average_repro {e[len("summary_repro "):]}' if 'summary_repro' in e else e for e in
         mAP_latex_rows], mAP_COLUMNS].rename(
        columns=rename_col_for_latex, index=DS_row_rename_dict)
    for name in simple_summary_names:
        summary_repro_df = df.loc[f'average_repro {name}'].apply(lambda x: str(round(x, 2))) + "±" + df.loc[
            f'stdev_repro {name}'].apply(
            lambda x: str(round(x, 2)))
        summary_repro_df.name = f'summary_repro {name}'

        DS_latex_df = DS_latex_df.append(summary_repro_df)

        if name == 'D':
            mAP_latex_df = df.append(summary_repro_df).loc[
                mAP_latex_rows, mAP_COLUMNS].rename(columns=rename_col_for_latex, index=DS_row_rename_dict)
    DS_latex_df = DS_latex_df.loc[
        DS_latex_rows, DS_COLUMNS].rename(columns=rename_col_for_latex, index=DS_row_rename_dict)
    print(DS_latex_df)
    old_DS_string = DS_latex_df.style.apply(b_g, cmap=USED_CM, source=color_source_DS_df).format(precision=2).to_latex(
        convert_css=True,
        column_format='llll|lll|l'
    )
    buf = io.StringIO(old_DS_string)
    lines = buf.readlines()
    newlines = []
    for l in lines:
        newlines.append(l)
        if any([v in l for v in [f'{VILBERT_LTX_NAME} reported', f'{VILBERT_LTX_NAME} repro', LITERAL_COPY_LTX_NAME]]):
            newlines.append("\\hline")

    newstr = "\n".join(newlines)
    print(newstr)

    # mAP_latex_df = df.append(summary_repro_df).loc[
    #     mAP_latex_rows, mAP_COLUMNS].rename(columns=rename_col_for_latex, index=DS_row_rename_dict)
    print(mAP_latex_df)
    color_fixed = mAP_latex_df.style.apply(b_g, cmap=USED_CM, source=color_source_mAP_df.astype(float)).format(
            precision=2).to_latex(convert_css=True)
    print(color_fixed)
    replacee = '{' + re.escape('\\') + 'cellcolor\[HTML\]{([0-9A-Z]+)}} ' + re.escape(
        '\\') + 'color{([a-z]+)} (-?[0-9,\.,±]+)'
    replacer = re.escape('\\') + 'multicolumn{1}{c}{{' + re.escape('\\') + r'cellcolor[HTML]{\1}} ' + re.escape(
        '\\') + r'color{\2} \3}'
    multicol_fixed = re.sub(replacee, replacer, color_fixed)
    print(multicol_fixed)

    # DS_style = color_source_DS_df.style.format(precision=2).background_gradient(cmap=cm, axis=0)
    # mAP_latex_df = df.loc[
    #     ['average_repro', 'literal_copy', 'devlbert_reported', 'vilbert', 'vilbert_reported', 'dependent_prior',
    #      'no_prior'], mAP_COLUMNS].rename(columns=rename_col_for_latex, index=DS_row_rename_dict)
    #
    # print(mAP_latex_df)
    # print(mAP_latex_df.style.format(precision=2).background_gradient(cmap=cm, axis=0).to_latex(
    #     convert_css=True))  # print(total_df)
    #     df = get_scores_for_run_name(r)
    #     mAP_scores.append(float(df['mAP_devlbert']))
    #     baseline_scores.append(float(df['mAP_baseline_emp']))
    # repro_score = sum(mAP_scores)/len(mAP_scores)
    # mAP_stdev = statistics.stdev(mAP_scores)
    # baseline_score = sum(baseline_scores)/len(baseline_scores)
    # baseline_stdev = statistics.stdev(baseline_scores)
    #
    # lc_df = get_scores_for_run_name(literal_run)
    # lc_score = float(lc_df['mAP_devlbert'])
    #
    # print(pd.DataFrame(list(zip(
    #     ('Literal copy', 'Reproduced run','Random baseline'),
    #     (lc_score, repro_score,baseline_score),
    #     (1,len(mAP_scores),len(mAP_scores)),
    #     ('/',mAP_stdev,baseline_stdev))),
    #     columns=("Type of run", "mAP score","# runs","standard deviation")).to_latex(index=False))


def print_avgAtt(avgAtt_dfs, c_avgAtt_dfs):
    AA_runs = [BEST_REPRO_RUN, 'dependent_prior', 'no_prior', 'literal_copy']
    multi = pd.concat({el.name: el for el in avgAtt_dfs}, axis=0)
    multi_c = pd.concat({el.name: el for el in c_avgAtt_dfs}, axis=0)
    multi_for_latex = multi.loc[multi.index.get_level_values(0).str.contains('|'.join(AA_runs))] \
        .rename(DS_row_rename_dict)
    c_multi_for_latex = multi_c.loc[multi_c.index.get_level_values(0).str.contains('|'.join(AA_runs))] \
        .rename(DS_row_rename_dict)
    og_string = str(multi_for_latex.style
        .apply(
        b_g, cmap=USED_CM,
        source=c_multi_for_latex,
        axis=None)
        .format(precision=2)
        .to_latex(
        column_format=f'p{{3cm}}l{"l" * NB_CAUSES}',
        convert_css=True,
        multirow_align='c'))


    buf = io.StringIO(og_string)
    lines = buf.readlines()
    newlines = []
    for i,l in enumerate(lines):
        if i == 1:
            # Put instead of index line
            l = "& Effect variable   & Top cause variables \\\\\n"
        if 'multirow' in l or 'end{tabular}' in l:
            newlines.append("\\hline\n")
        newlines.append(l)

    new_string = "".join(newlines)
    new_string = new_string.replace("\multirow[c]{10}{*}", "\multirow[c]{10}{3cm}")
    print(new_string)


if __name__ == '__main__':
    main()
