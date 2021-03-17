import json
FOR_REAL = False
from time import time
TEST_READY = True
SANDBOX = True
from easyturk import EasyTurk

et = EasyTurk(sandbox=SANDBOX)
client = et.mtc
with open('questionForm.xml','r') as f:
    question_form_string = ''.join(f.readlines())
with open('answerKey.xml','r') as f:
    answer_key_string = ''.join(f.readlines())
if TEST_READY:
    response = client.create_qualification_type(
        Name=f'Test_{round(time())}' if not FOR_REAL else 'Qualification test',
        Keywords='test',
        Description='Test with examples of causation between objects in scene, to ensure the worker correctly understand what exactly is meant.',
        QualificationTypeStatus='Active',
        RetryDelayInSeconds=3600*24*7,
        Test=question_form_string,
        AnswerKey=answer_key_string,
        TestDurationInSeconds=60*15,
        AutoGranted=False
    )
    response_string = json.dumps(response, indent=4, sort_keys=True, default=str)
    with open(f"test_qualification_response{'_sandbox' if SANDBOX else ''}.json", "w") as f:
        f.write(response_string)
