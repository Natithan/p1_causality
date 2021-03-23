from typing import Dict, List

import pandas as pd
from pathlib import Path
from constants import MTURK_DIR
import json
import re
from collections import Counter
from itertools import combinations
from collections import defaultdict

datas = {}
input_dir = Path(MTURK_DIR, 'input_mturk')
RESPONSES_FILE = Path(MTURK_DIR, 'output_mturk/results_for_pair_2021_03_22__15_03_14.tsv')


def clean_up(x: str) -> str:  # to deal with e.g. '(\'outfield\', "pitcher\'s mound"), or 'None', or ""?""
    cleaned_up = re.sub(r'(?<=\w)\'(?=\w)', '@@TMP@@', x) \
        .replace('\'', '\"') \
        .replace('\\\'', '\"') \
        .replace('\"\"?\"\"', '\"?\"') \
        .replace('(', '[') \
        .replace(')', ']') \
        .replace('@@TMP@@', '\'') \
        .replace('None', 'null')
    return cleaned_up


def max_response(row, w_for_cfdnce_lvl: Dict[str, List[int]]):
    '''

    Returns: most frequent response + its frequency as a fraction of maximum frequency

    '''
    c = Counter()
    for k, cfdnce_lvl in zip(row['cause_directions'], row['confidences']):
        c.update({k: w_for_cfdnce_lvl[cfdnce_lvl]})
    max_resp, count = sorted(c.items(), key=lambda item: item[1], reverse=True)[0]
    count_fraction = count / len(row['cause_directions'])
    return max_resp, count_fraction, count


def main():
    responses = pd.read_csv(RESPONSES_FILE,
                            sep='\t',
                            converters={col: lambda x: json.loads(clean_up(x))
                                        for col in ['cause_directions', 'confounders', 'confidences']}
                            )
    responses[['word_X', 'word_Y']] = pd.DataFrame([p.split(" // ") for p in responses['Unnamed: 0'].tolist()],
                                                   index=responses.index)
    w_for_cfdnce_setting = {
        "no_cfdnce_weight": [1, 1, 1],
        "half_cfdnce_weight": [.5, .75, 1],
        "full_cfdnce_weight": [1 / 3, 2 / 3, 3 / 3],
    }
    for min_agreement in [.6, .8, 1]:
        for cfdnce_setting, cfdnce_weighting in w_for_cfdnce_setting.items():
            w_for_cfdnce_lvl = {
                "confidence_1": cfdnce_weighting[0],
                "confidence_2": cfdnce_weighting[1],
                "confidence_3": cfdnce_weighting[2],
                None: 0, # Ignore some cases where input is None
            }
            responses[['max_resp', 'max_resp_fraction', 'max_resp_count']] = pd.DataFrame(
                [max_response(row, w_for_cfdnce_lvl) for _, row in responses.iterrows()], index=responses.index)
            filtered_responses = \
                responses[responses.apply(lambda row:
                                          row['max_resp_fraction'] >= min_agreement and
                                          row['max_resp'] != 'x-is-y' and
                                          len(row['cause_directions']) >= 5,
                                                                               axis=1)]
            cause_only_responses = filtered_responses[filtered_responses.apply(lambda row:
                                                                               row['max_resp'] in ['x-to-y', 'y-to-x'],
                                                                               axis=1)]
            x_to_y = [(a.word_X, a.word_Y) if a.max_resp == 'x-to-y' else (a.word_Y, a.word_X) for a in
                      cause_only_responses.itertuples(index=False)]
            ys_for_x = defaultdict(list)
            for x, y in x_to_y:
                ys_for_x[x] += [y]

            confnd_triples = [f"{eff1}⬅{cause}➡{eff2}" for cause in ys_for_x for eff1, eff2 in
                              combinations(ys_for_x[cause], 2) if
                              len(ys_for_x[cause]) > 1]

            dir = Path(MTURK_DIR,'output_mturk')

            # storing pairs
            pair_file_id = f'pair_annotations_{min_agreement}_{cfdnce_setting}'
            pair_file_path = Path(dir, f'{pair_file_id}.tsv')
            filtered_responses[['word_X', 'word_Y', 'max_resp']].to_csv(pair_file_path, sep='\t', index=False)

            # storing triplets
            conf_file_id = f'conf_triples_{min_agreement}_{cfdnce_setting}'
            with open(Path(dir, f'{conf_file_id}.txt'), 'w') as f:
                f.truncate(0)
                for t in confnd_triples:
                    f.write(t + "\r\n")




if __name__ == '__main__':
    main()

# for ranking in ['absolute', 'relative']:
#     modality = 'image'
#
#     filter = ''
#     if ranking == 'relative':
#         filter = '_positive'
#     file = Path(input_dir, f"{ranking}_{modality}{filter}.tsv")
#     df = pd.read_csv(file,
#                      sep='\t',
#                      converters={col: lambda x: json.loads(
#                          clean_up(x))
#                                  for col in ['words', 'joint_image_ids', 'marginal_img_ids']})
#     datas[f"{ranking}_{modality}{filter}"] = df
#
# df1 = datas['absolute_image']
# df2 = datas['relative_image_positive']
# for df, typ in zip([df1, df2], ['absolute', 'relative']):
#     pairs = df['words'].values
#     pairs = [sorted(l) for l in pairs]
#     vals = sorted(list(set([i for (i, j) in pairs] + [j for (i, j) in pairs])))
#     triples = [(i, j, k) for i, j in pairs
#                for k in vals
#                if ((([j, k] in pairs) or ([k,j] in pairs)) and
#                    (([i, k] in pairs) or ([k,i] in pairs)))]
#     with open(Path(MTURK_DIR, f'triplets_{typ}.tsv'), 'w') as f:
#         f.truncate(0)
#         for t in triples:
#             f.write('\t'.join(str(v) for v in t) + "\r\n")
