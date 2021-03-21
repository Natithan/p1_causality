from easyturk import EasyTurk
import re

from mturk.mturk_util import ALL_HIDS


def main():
    et = EasyTurk(sandbox=False)
    cl = et.mtc
    print_submit_times(cl)

def print_submit_times(cl):
    datetimes = sorted([a['SubmitTime'] for hid in ALL_HIDS for a in cl.list_assignments_for_hit(HITId=hid)['Assignments'] if
                        (cl.list_assignments_for_hit(HITId=hid)['NumResults'] > 0)])
    times =[(t.day,t.hour) for t in datetimes]
    counts = sorted(set([(t,times.count(t)) for t in times]))
    current_day = -1
    for (day,hour),count in counts:
        string = ''
        if current_day != day:
            current_day = day
            string += f"Day: {current_day}\t\t "
        else:
            string += "\t\t\t "
        string += f"Hour: {hour}: {count} submissions"
        print(string)
    print(f"Total submits: {len(datetimes)} / {len(ALL_HIDS) * 5} = {100 * len(datetimes) / (len(ALL_HIDS) * 5)}%")
    results_per_worker = get_assignment_results_per_worker(cl)
    agg = [(wid,round(100*len(resp)/sum(len(v) for v in results_per_worker.values()))) for wid,resp in results_per_worker.items()]
    [print(a) for a in agg]

def get_remaining_for_pair(cl):

    all_hits = [cl.get_hit(HITId=hid)['HIT'] for hid in ALL_HIDS]
    max_ass = all_hits[0]['MaxAssignments']
    assignments = [cl.list_assignments_for_hit(HITId=hid)['Assignments'] for hid in ALL_HIDS]
    answer_regex = r'"marginal_url_y":".+?","word_X":"(.+?)","word_Y":"(.+?)"'
    pair_answers = [pair for a in assignments for d in a for pair in re.findall(answer_regex, d['Answer'])]
    remaining_for_pair = {p: max_ass - pair_answers.count(p) for p in set(pair_answers)}
    return remaining_for_pair


def get_assignment_results_per_worker(cl):
    assignments = [a for hid in ALL_HIDS for a in cl.list_assignments_for_hit(HITId=hid)['Assignments']]

    answer_regex = r'"marginal_url_y":".+?","word_X":"(.+?)","word_Y":"(.+?)","cause_directions":"(.+?)"'
    aa = [(a['WorkerId'], a['AssignmentStatus'], tripl) for a in assignments for tripl in
          re.findall(answer_regex, a['Answer'])]
    dic = {}
    for wid, ass_status, pair in aa:
        if wid in dic:
            dic[wid] += [(pair, ass_status)]
        else:
            dic[wid] = [(pair, ass_status)]
    return dic

if __name__ == '__main__':
    main()
