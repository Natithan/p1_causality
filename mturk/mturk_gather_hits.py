import json

from pathlib import Path
import pandas as pd
import sys
import datetime
from absl import flags
from constants import ROOT_DIR, MTURK_DIR

import interface
from easyturk import EasyTurk
# ALL_HIDS = ['3DW3BNF1HOC4KK5Z8KVPGY22Z89V8N', '3AA88CN99WX5JF25MKFCD2FNCP5KY4', '3DIIW4IV9WPYFDYXXA6TKXW3PKHI46', '3OID399FYN1R5IF0PD11K4DHM5IFDP', '3VLL1PIEOXIOEWXPGY9OYQ5Y2OJZOU', '3909MD9T36BWXOT0L8CLQMV2MHJFEI', '3QQUBC640L8HRGXYE1LQMGPI4EGXNG', '35A1YQPVGLAGBW2XBFP3PP2HPGCI5B', '3IVKZBIBK73HL3POHTKQIOBU3N5SHU', '3KG2UQJ0NQIXIJWMSNQYS8TP7BTNQ9', '3Z33IC0JD7G4YPO9DC13GUXDJERV9C', '3RWSQDNYMGGO32TWGJWR84G2YIYFF0', '3MVY4USGCDHHHY08NB2BI0C930KSIN', '3NRZ1LDP8306ZW9QVYM3NQAOCTFZPB', '363A7XIFWBGI38YE5IF3CP8XPW9VAR', '3M93N4X8IRHDC9KE5TNBU1R9GF9SJZ', '3YGYP136583XUKBM52FYXY3U3XVNRA', '3VI0PC2ZB5EZ91XAFGL315PNUOLXOV', '32XN26MTY6DJ6QO2ZCG40AYIPCVL07', '3YCT0L9ONT39VZBZKRF3N88QPZENSY', '3NFWQRSHWL8UH2PRR329QPGEZXYFGZ', '3XT3KXP256SCU02BGYZ8HVGHLBNI6R', '3XABXM4AK8ZCFC9IML1FAB3UIGUQ8O', '3NOEP8XAVBWHU19BUGYIQ5UD4THXPC', '30EMX9PEWRD8NTQWNENNVGRMV49SKY', '3GVPRXWRQOOVY0GGZ840N9GDGM7I7Q', '31J7RYEC0SKJDF9KVVK4EFHSU41L1A', '3T5ZXGO9ELIRZ8A5VB1SNV0YHP1ZQ2', '3JHB4BPSGR3CTHSSRD7TA7Y52MCQ9D', '3T2EL38U1TE2UDAQUTD7QAKN9P3XQ3', '35XW21VSWN8J7Z8E8EZOAG41K4ESL1', '36AZSFEY0BUNMI36BWFZ70S37EUVBM', '3N3WJQXEMZKQ20RGEZKIJYRXMAFL2R', '375VSR8FW33RB9P58QQSSLA3DB3ZR3', '372AGES0JBZ26EPQ7827V0US5B5XR4', '3O71U79SSIJ3DHQZ8Q03ATJQK94SM3', '371Q3BEXEO33COPINFQXIVFZZDMZSR', '3IHWR4LC8K7ZYELZ9DW619C8R6DI8D', '3YCT0L9ONT39VZBZKRF3N88QPZESN3', '3QHITW7OZV3QY02XJJLT629P84UQAS', '37Y5RYYI1WZE7TP3MX2CLAZORDOXSS', '36JW4WBR1DE8HYK2B9KRBPSFEXBFHY', '3DZKABX20PZ6WNVCPWBUIKYLPZUVCT', '3QO7EE373VHVHA7PPXLP1DTVQMFQBN', '32K26U12EUIG1F1QPS652QG3AZ6VDT', '334ZEL5JYD9RDKBBL6FG2X8VF9JSOI', '3AJA9FLWTJS8DTVMHR2CB1TUEAQFIR', '3XDSWAMB39MERFZV3XHKCXZD87FQCU', '34OWYT6U43BZCJ49E52K157JBCVI92', '3N2YPY1GJDSLTLFQLNHP88YOAB7VEM', '3VADEH0UIJR9YKNC06SVRXDLPEFSPZ', '3U74KRR68TFAAZOE9RKTXD4CRISNTN', '3MYASTQBHE5OW7593TCVW3HVT7RQDU', '33W1NHWFZOF4RO2XCFVNS0BL1W0ZTG', '3BO3NEOQN7BDH2EE6BGKX0I3HUDIAH', '31ODACBEO19UYWOR0J7KR23VUA1SQQ', '3AXFSPQOZXS484FSZ9NCN28URPFFJ3', '37G6BXQPMXFTODJ9ZONF2LZGTJSQEN', '3Y7LTZE0Z0GJR1HMCKLXXWXNKA7ZU7', '3O2Y2UIUDXOZJOLAHUNOOH876EFFK2', '37SOB9Z0TZRFNINZIZYN28WPSOFL32', '3BVS8WK9R7PI0CJ6CPGGSB29ZCYIBC', '3FO95NVK6JUUAX3RDYWKWSD0QW3SRR', '3PCPFX4U57KLURJ5UZ7LK3KG5K7QF5', '3WKGUBL7T6G05E63ID36CDOV6OBL4N', '3IKDQS3DRLU1AHBCQPCB3V8RHXYICJ', '39RRBHZ0B1VJ4G6MHAV2TIZ6YHGZVW', '3PIOQ99R85GPAC339WA329QEAWZNUE', '3LVTFB9DFCCR8RF05JD32OKS6Z7QG4', '33TGB4G0MWBFMT2IBX72VFVATW2XTH', '3TX9T2ZCCGVWJ56R7K0YFKIK5QIZW6', '3TL87MO8DTJA3U3S2UZP3HLMVEKFL5', '3OCZWXS70V1I1XA3WIMGH5U96K6L5S', '375VMB7D5QDBF9HQQL7MN1Q92XAIDJ', '302U8RURK6VPNRS3EMK8YVSXO38NV3', '3CVDZS289OUU9CLD26043U0BVJAFM7', '34ZTTGSNK4I58NABPPVLNOWTLZKQH3', '35YHTYFL2NXDZZFMGY1VQQJOMCMVF4', '3GV1I4SEPGJEK1AH11WL9B892FHL68', '3ZFRE2BDRG8B85BHHPWKH3WRESOZXZ', '3K8CQCU3LLV22GS84WP4KXBBVCANWD', '3M0556244ZE0RU6DE7F4G9PB09KFN7', '38Z7YZ2SCAW54ILVV7D6N0X8LCZQIW', '3ZXNP4Z3AYFXO1OMKB1DFP85XQ1L77', '3FK4G712O4U6BC34SNWPM2IWCYMSSF', '3RSBJ6YZFJKI9F6PFMFHVYPGQJPFOM', '38DCH97KIOW1ZT51DPY6Z1C8YROQJ8', '31HLTCK4CSPJDZBHRI7D8BJ0NRMVG3', '31JUPBOOSUY1OFT5UGTJTP408A7L8U', '3HXCEECSRTN08A1RMLIM0H30GYIZY5', '3SNR5F7RA9NG7FVQMGI6TJ8U29BIEC', '3QQUBC640L8HRGXYE1LQMGPI4EGNX6', '3PCPFX4U57KLURJ5UZ7LK3KG5K7FQU', '3R15W654WKN7UJT1GAAJFGP02GTQLA', '3NQUW096OD212KCFZ8ZXTLZBSGPL9J', '3ZICQFRS48C0D0BWI7K5EOC2MSMZZW', '3BFF0DJK9467QCGJHN1FW7EIEH0ST4', '3AJA9FLWTJS8DTVMHR2CB1TUEAQIFU', '3ABAOCJ4SFYR01BMGMBYFT4P2LJQMC', '3KLL7H3EHKVXDV6SBOPVTBV12RZVH2', '36QZ6V159G7MQPV8HSRP130KXV7SUV', '362E9TQF3OKERTRHSB8UTMT6FPQIGT', '3KG2UQJ0NQIXIJWMSNQYS8TP7BTQNC', '388FBO7J0YN6RLN8JX7S5UWR6KANYC', '33CLA8O0NP5L6SY57EWLPTUL169FRV', '39KMGHJ4S6469UZGAJUL2Q6UMEL00E', '3QTFNPMJDDC6WBXDFJ9BJ15TCEENZ3', '3HA5ODM5LHMM34K8MI1UXP23B2GSVK', '304QEQWK0WERCBRZC4WWSR2CKH0O0R', '31GN6YMHMWMZITKDCS6QJRLHIBISWU', '3OLZC0DJ9Q9X9QHCH67GTNWG24EVIV', '386659BNUSB6GJKY62YLGVP4R6R01H']
from mturk.mturk_util import ALL_HIDS
# region Flags stuff
FGS = flags.FLAGS
flags.DEFINE_bool("sandbox", False, "")
FGS(sys.argv)

def main():

    results = interface.fetch_completed_hits(ALL_HIDS, approve=False, sandbox=FGS.sandbox)
    output_dir = Path(MTURK_DIR, 'output_mturk')
    string_results = json.dumps(results, indent=4, sort_keys=True, default=str)
    with open(Path(output_dir, f'results_{datetime.datetime.now().strftime("%Y_%m_%d__%H_%m_%S")}.json'), 'a') as f:
        f.write(string_results)

    res_for_pair = get_results_for_pair(results)

    df = pd.DataFrame(res_for_pair).transpose()
    with open(Path(output_dir, f'results_for_pair_{datetime.datetime.now().strftime("%Y_%m_%d__%H_%m_%S")}.tsv'), 'a') as f:
        f.write(df.to_csv(sep='\t'))


def get_results_for_pair(results):
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
    return res_for_pair


if __name__ == '__main__':
    main()

# from easyturk import EasyTurk
# import re
#
# from mturk.mturk_util import ALL_HIDS
#
# et = EasyTurk(sandbox=False)
# cl = et.mtc
#
#
# def get_remaining_for_pair(cl):
#     all_hits = [cl.get_hit(HITId=hid)['HIT'] for hid in ALL_HIDS]
#     max_ass = all_hits[0]['MaxAssignments']
#     assignments = [cl.list_assignments_for_hit(HITId=hid)['Assignments'] for hid in ALL_HIDS]
#     answer_regex = r'"marginal_url_y":".+?","word_X":"(.+?)","word_Y":"(.+?)"'
#     pair_answers = [pair for a in assignments for d in a for pair in re.findall(answer_regex, d['Answer'])]
#     remaining_for_pair = {p: max_ass - pair_answers.count(p) for p in set(pair_answers)}
#     return remaining_for_pair
#
#
# def get_assignment_results_per_worker(cl):
#     assignments = [a for hid in ALL_HIDS for a in cl.list_assignments_for_hit(HITId=hid)['Assignments']]
#
#     answer_regex = r'"marginal_url_y":".+?","word_X":"(.+?)","word_Y":"(.+?)","cause_directions":"(.+?)"'
#     aa = [(a['WorkerId'], a['AssignmentStatus'], tripl) for a in assignments for tripl in
#           re.findall(answer_regex, a['Answer'])]
#     dic = {}
#     for wid, ass_status, pair in aa:
#         if wid in dic:
#             dic[wid] += [(pair, ass_status)]
#         else:
#             dic[wid] = [(pair, ass_status)]
#     return dic
#
#
# afw = get_assignment_results_per_worker(cl)
# rfp = {k: v for k, v in get_remaining_for_pair(cl).items() if v > 0}
# from mturk.mturk_util import ALL_HIDS
#
# assignments = [a for hid in ALL_HIDS for a in cl.list_assignments_for_hit(HITId=hid)['Assignments']]
#
# answer_regex = r'"marginal_url_y":".+?","word_X":"(.+?)","word_Y":"(.+?)","cause_directions":"(.+?)","confounders":"(.+?)","confidences":"confidence_(.+?)"'
# aa = [(a['WorkerId'], a['AssignmentStatus'], tpl) for a in assignments for tpl in
#       re.findall(answer_regex, a['Answer'])]
# dic = {}
# for wid, ass_status, tpl in aa:
#     if wid in dic:
#         dic[wid] += [(tpl, ass_status)]
#     else:
#         dic[wid] = [(tpl, ass_status)]
#
# answer_regex = r'"marginal_url_y":".+?","word_X":"(.+?)","word_Y":"(.+?)","cause_directions":"(.+?)","confounders":"(.+?)","confidences":"confidence_(.+?)"}'
# aa = [(a['WorkerId'], a['AssignmentStatus'], tpl) for a in assignments for tpl in
#       re.findall(answer_regex, a['Answer'])]
# dic = {}
# for wid, ass_status, tpl in aa:
#     if wid in dic:
#         dic[wid] += [(tpl, ass_status)]
#     else:
#         dic[wid] = [(tpl, ass_status)]
#
# answer_regex = r'"marginal_url_y":".+?","word_X":"(.+?)","word_Y":"(.+?)","cause_directions":"(.+?)","confounders":"(.*?)","confidences":"confidence_(.+?)"}'
# aa = [(a['WorkerId'], a['AssignmentStatus'], tpl) for a in assignments for tpl in
#       re.findall(answer_regex, a['Answer'])]
# dic = {}
# for wid, ass_status, tpl in aa:
#     if wid in dic:
#         dic[wid] += [(tpl, ass_status)]
#     else:
#         dic[wid] = [(tpl, ass_status)]
#
# for wid in dic:
#     print(wid)
#
# import random
#
# for wid in dic:
#     print(wid)
#     print(random.sample(dic[wid]), 10)
#
# for wid in dic:
#     print(wid)
#     print(random.sample(dic[wid], 10))
#
# for wid in dic:
#     print(wid)
#     if len(dic[wid]) > 10:
#         print(random.sample(dic[wid], 10))
#     else:
#         print(dic[wid])
#
# OK_LIST = [
#     'AVIEE6LDH0BT5', 'AE3ZQZ2Z81Z11'
# ]
# OK_LIST = [
#     'AVIEE6LDH0BT5', 'AE3ZQZ2Z81Z11', 'AKQAI78JTXXC9', 'A3UWHJTAUTF6TO', 'A2ZP67F7PDFFLF', ''
# ]
# rfp[('pillows', 'player')]
# rfp[('player', 'pillows')]
# [v[i] for _, v in afw.keys() for i in v if (('player' in v[i][0]) and ('pillow' in v[i][0]))]
# [v[i] for v in afw.values() for i in v if (('player' in v[i][0]) and ('pillow' in v[i][0]))]
# [i for v in afw.values() for i in v if (('player' in v[i][0]) and ('pillow' in v[i][0]))]
# [v[i] for v in afw.values() for i in v if (('player' in i[0]) and ('pillow' in i[0]))]
# [i for v in afw.values() for i in v if (('player' in i[0]) and ('pillow' in i[0]))]
# [i for v in afw.values() for i in v if (('player' in i[0]))]
# [i for v in afw.values() for i in v if (('player' in i[0]) and ('pillows' in i[0]))]
# [i for v in dic.values() for i in v if (('player' in i[0]) and ('pillows' in i[0]))]
# for wid in dic:
#     print(wid, len(dic[wid]))
#     if len(dic[wid]) > 10:
#         print(random.sample(dic[wid], 10))
#     else:
#         print(dic[wid])
#
# OK_LIST = [
#     'AVIEE6LDH0BT5', 'AE3ZQZ2Z81Z11', 'AKQAI78JTXXC9', 'A3UWHJTAUTF6TO', 'A2ZP67F7PDFFLF', ''
# ]
# DOUBT_LIST = [
#
# ]
#
# OK_LIST = [
#     'AVIEE6LDH0BT5', 'AE3ZQZ2Z81Z11', 'AKQAI78JTXXC9', 'A3UWHJTAUTF6TO', 'A2ZP67F7PDFFLF', ''
# ]
# DOUBT_LIST = [
#     'A1SX8IVV82M0LW'
# ]
# ALREADY_APPROVED_LIST = [
#     'A3W2RRAUPHYRIO'
# ]
# for wid in dic:
#     non_approved = [a for a in dic[wid] if a[1] != 'Approved']
#     print(wid, len(dic[wid]), len(non_approved))
#     if len(non_approved) > 10:
#         print(random.sample(dic[wid], 10))
#     else:
#         print(non_approved)
#
# for wid in dic:
#     non_approved = [a for a in dic[wid] if a[1] != 'Approved']
#     print(wid, len(dic[wid]), len(non_approved))
#     if len(non_approved) > 10:
#         print(random.sample(non_approved, 10))
#     else:
#         print(non_approved)
#
# OK_LIST = [
#     'AVIEE6LDH0BT5', 'AE3ZQZ2Z81Z11', 'AKQAI78JTXXC9', 'A3UWHJTAUTF6TO', 'A2ZP67F7PDFFLF', 'AX7K5UODLEK72',
#     'A1RF95TSDZDJLL', 'A16U1L4R6WV5G2', 'A11HTUAZMFBEO7',
#     'A2QX3YJXAAHHVV', 'A1LLW7JQ0IECHQ', 'A3AOE0Y5VK6JYF', 'AYHIH9NTPYFLY', 'AHXHM1PQTRWIQ', 'A26RIX88SGQT0S',
#     'A1V8FJHR0XVNCV', 'AQ6P4JFUUXXN9', 'ALKQPW0O9C98N', 'A1BQ71W4QHT62P'
# ]
# DOUBT_LIST = [
#     'A1SX8IVV82M0LW', 'A2SXYRKIP3J9CJ', 'A3HJNOCKXFOOVP'
# ]
# ALREADY_APPROVED_LIST = [
#     'A3W2RRAUPHYRIO', 'AEBY6S5MX5O8R', 'A302KOFOYLD89A', 'A2RWD6C4G4RMW6', 'A28IR8BBKC2CQJ'
# ]
# for wid in dic:
#     if not wid in DOUBT_LIST:
#         continue
#     non_approved = [a for a in dic[wid] if a[1] != 'Approved']
#     print(wid, len(dic[wid]), len(non_approved))
#     if len(non_approved) > 10:
#         print(random.sample(non_approved, 10))
#     else:
#         print(non_approved)
#
# OK_LIST.append('A1SX8IVV82M0LW')
# DOUBT_LIST = [
#     'A2SXYRKIP3J9CJ', 'A3HJNOCKXFOOVP'
# ]
# REJECT_LIST = DOUBT_LIST
# approve_ass_ids = [a['AssignmentId'] for a in assignments if a['WorkerId'] in OK_LIST]
# reject_ass_ids = [a['AssignmentId'] for a in assignments if a['WorkerId'] in REJECT_LIST]
# len([a['AssignmentId'] for a in assignments if a['AssignmentStatus'] != 'Submitted'])
# approve_ass_ids = [a['AssignmentId'] for a in assignments if
#                    a['WorkerId'] in OK_LIST and a['AssignmentStatus'] == 'Submitted']
# reject_ass_ids = [a['AssignmentId'] for a in assignments if
#                   a['WorkerId'] in REJECT_LIST and a['AssignmentStatus'] == 'Submitted']
# for app in approve_ass_ids:
#     cl.approve_assignment(AssignmentId=app, RequesterFeedback='Well done!')
#
# A2SXYRKIP3J9CJ_ass = [a['AssignmentId'] for a in assignments if
#                       a['WorkerId'] == 'A2SXYRKIP3J9CJ' and a['AssignmentStatus'] == 'Submitted']