from pathlib import Path

from constants import MTURK_DIR
from easyturk import EasyTurk
from mturk.mturk_util import ALL_HIDS
from util import TIME
import json

SANDBOX = False
TEST_PERCENTAGE_MIN = 90
PAST_ACCEPTANCE_PERCENTAGE_MIN = 95
NB_HITS_MIN = 50
DATA_PER_HIT = 9
TITLE = f'Say whether one object in a scene "causes" another{TIME if SANDBOX else ""}'
DESCRIPTION = f'Say whether you think intervening on the presence of an object in a scene would have' \
              f' consequences on the probability of the presence of another object.' \
              f'Takes a few minutes to check the explanation video and pass the qualification test,' \
              f'and then each hit asks you to label {DATA_PER_HIT} pairs, ' \
              f'where each pair probably takes about 10 seconds.'
KEYWORDS = 'causation, image, objects'
TIME_PER_HIT = 60*60  # One hour time
AUTO_APPROVAL_DELAY = 60*60*24*7 # One week time before auto-approving
REWARD = str(0.4)


def update_hit_types(cl,hitTypeId):
    for hid in ALL_HIDS:
        cl.update_hit_type_of_hit(
            HITId=hid,
            HITTypeId=hitTypeId
        )


def main():
    et = EasyTurk(sandbox=SANDBOX)
    cl = et.mtc
    response = create_hit_type(cl)
    hitTypeId = response['HITTypeId']
    update_hit_types(cl,hitTypeId)


def create_hit_type(cl):
    with open(Path(MTURK_DIR, f"test_qualification_response{'_sandbox' if SANDBOX else ''}.json"), "r") as f:
        response = json.loads(''.join(f.readlines()))
        test_qual_id = response['QualificationType']['QualificationTypeId']
    hit_type_args = {
        'Title': TITLE,
        'Description': DESCRIPTION,
        'Keywords': KEYWORDS,
        'AssignmentDurationInSeconds': TIME_PER_HIT,
        'AutoApprovalDelayInSeconds': AUTO_APPROVAL_DELAY,
        'QualificationRequirements': [
            {
                'QualificationTypeId': '00000000000000000040',
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues': [NB_HITS_MIN]
            },
            # {
            #     'QualificationTypeId': '00000000000000000071',
            #     'Comparator': 'EqualTo' if len(countries) == 1 else 'In',
            #     'LocaleValues': [
            #          {'Country': country} for country in countries
            #     ],
            # },
            {
                'QualificationTypeId': '000000000000000000L0',
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues': [PAST_ACCEPTANCE_PERCENTAGE_MIN],
            },
            {
                'QualificationTypeId': test_qual_id,
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues': [TEST_PERCENTAGE_MIN]
            }
        ],
        'Reward': REWARD
    }
    response = cl.create_hit_type(**hit_type_args)
    return response


if __name__ == '__main__':
    main()
