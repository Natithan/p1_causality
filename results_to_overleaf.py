import os, glob
import pandas as pd
import num
import statistics
from constants import PROJECT_ROOT_DIR
repro_runs = ('gimli_1','gimli_2','v4','v5','v6')
literal_run = 'literal_copy'


def get_scores_for_run_name(r):
    files = glob.glob(f'{PROJECT_ROOT_DIR}/PP_output/{r}/mAP_output/avg_mAP_comparison_*_90760*.csv')
    assert files, f"No match found for {PROJECT_ROOT_DIR}/PP_output/{r}/mAP_output/avg_mAP_comparison_*_90760*.csv"
    return pd.read_csv(files[0])

def main():
    mAP_scores = []
    baseline_scores = []
    for r in repro_runs:
        df = get_scores_for_run_name(r)
        mAP_scores.append(float(df['mAP_devlbert']))
        baseline_scores.append(float(df['mAP_baseline_emp']))
    repro_score = sum(mAP_scores)/len(mAP_scores)
    mAP_stdev = statistics.stdev(mAP_scores)
    baseline_score = sum(baseline_scores)/len(baseline_scores)
    baseline_stdev = statistics.stdev(baseline_scores)

    lc_df = get_scores_for_run_name(literal_run)
    lc_score = float(lc_df['mAP_devlbert'])

    print(pd.DataFrame(list(zip(
        ('Literal copy', 'Reproduced run','Random baseline'),
        (lc_score, repro_score,baseline_score),
        (1,len(mAP_scores),len(mAP_scores)),
        ('/',mAP_stdev,baseline_stdev))),
        columns=("Type of run", "mAP score","# runs","standard deviation")).to_latex(index=False))


if __name__ == '__main__':
    main()