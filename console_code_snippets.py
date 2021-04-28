import pandas as pd

df = pd.read_csv("/cw/liir/NoCsBack/testliir/nathan/p1_causality/mAP_output/per_class_mAP_comparison_1618598428_30278_rank0_my_devlbert.csv")
sum(df['count'])

df_responses = pd.read_csv("/cw/liir/NoCsBack/testliir/nathan/p1_causality/mturk/output_mturk/results_for_pair_2021_03_22__15_03_14.tsv",sep='\t')
import ast
filtered_df = df_responses[df_responses.apply(lambda x: len(ast.literal_eval(x['cause_directions'])) == 5, axis=1)]
remainder_df  = df_responses[df_responses.apply(lambda x: len(ast.literal_eval(x['cause_directions'])) != 5, axis=1)]