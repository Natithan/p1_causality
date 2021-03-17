import pandas as pd
from pathlib import Path
from constants import MTURK_DIR
import json
import re
datas = {}
input_dir = Path(MTURK_DIR, 'input_mturk')
for ranking in ['absolute', 'relative']:
    modality = 'image'

    filter = ''
    if ranking == 'relative':
        filter = '_positive'
    file = Path(input_dir, f"{ranking}_{modality}{filter}.tsv")
    df = pd.read_csv(file,
                     sep='\t',
                     converters={col: lambda x: json.loads(
                         re.sub(r'(?<=\w)\'(?=\w)', '@@TMP@@', x) # to deal with e.g. '(\'outfield\', "pitcher\'s mound")'
                             .replace('\'', '\"')
                             .replace('\\\'', '\"')
                             .replace('(', '[')
                             .replace(')', ']')
                             .replace('@@TMP@@', '\''))
                                 for col in ['words', 'joint_image_ids', 'marginal_img_ids']})
    datas[f"{ranking}_{modality}{filter}"] = df

df1 = datas['absolute_image']
df2 = datas['relative_image_positive']
for df, typ in zip([df1, df2], ['absolute', 'relative']):
    pairs = df['words'].values
    pairs = [sorted(l) for l in pairs]
    vals = sorted(list(set([i for (i, j) in pairs] + [j for (i, j) in pairs])))
    triples = [(i, j, k) for i, j in pairs
               for k in vals
               if ((([j, k] in pairs) or ([j, k] in pairs)) and
                   (([i, k] in pairs) or ([i, k] in pairs)))]
    with open(Path(MTURK_DIR, f'triplets_{typ}.tsv'), 'w') as f:
        f.truncate(0)
        for t in triples:
            f.write('\t'.join(str(v) for v in t) + "\r\n")

