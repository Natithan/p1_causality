import json

from pathlib import Path
import pandas as pd
import sys
import datetime
from absl import flags
from constants import ROOT_DIR, MTURK_DIR

import interface
from easyturk import EasyTurk

# region Flags stuff
FGS = flags.FLAGS
flags.DEFINE_bool("sandbox", False, "")
flags.DEFINE_integer("max_assignments", 5, "")
flags.DEFINE_float("reward", .5, "")
FGS(sys.argv)

def main():

    et = EasyTurk(sandbox=FGS.sandbox)
    with open(Path(MTURK_DIR, f"test_qualification_response{'_sandbox' if FGS.sandbox else ''}.json"), "r") as f:
        response = json.loads(''.join(f.readlines()))
        test_qual_id = response['QualificationType']['QualificationTypeId']


    # hit_ids = [h['HITId'] for h in et.mtc.list_reviewable_hits()['HITs']]
    hit_ids = [h['HITId'] for h in et.list_hits()]
    results = interface.fetch_completed_hits(hit_ids, approve=False, sandbox=FGS.sandbox)
    output_dir = Path(MTURK_DIR, 'output_mturk')
    string_results = json.dumps(results, indent=4, sort_keys=True, default=str)
    with open(Path(output_dir, f'results_{datetime.datetime.now().strftime("%Y_%m_%d__%H_%m_%S")}.json'), 'a') as f:
        f.write(string_results)

    res_for_pair = {}
    for hid in results:
        for assignment in results[hid]:
            os = assignment['output']
            wid = assignment['worker_id']
            for o in os:
                pair = " // ".join([o['word_X'], o['word_Y']])
                if pair in res_for_pair:
                    res_for_pair[pair]['workerd_id'].append(wid)
                    for k, v in o.items():
                        if k not in ('word_X', 'word_Y'):
                            res_for_pair[pair][k].append(v)
                else:
                    res_for_pair[pair] = {k: [v] for k, v in o.items() if k not in ('word_X', 'word_Y')}
                    res_for_pair[pair]['workerd_id'] = [wid]

    df = pd.DataFrame(res_for_pair).transpose()
    with open(Path(output_dir, f'results_for_pair_{datetime.datetime.now().strftime("%Y_%m_%d__%H_%m_%S")}.tsv'), 'a') as f:
        f.write(df.to_csv(sep='\t'))

if __name__ == '__main__':
    main()
