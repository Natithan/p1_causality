#!/usr/bin/python
import xml.dom.minidom
import os
import sys
import string
import time
import re
import argparse

f=os.popen('qstat --xml')
dom=xml.dom.minidom.parse(f)
jobs=dom.getElementsByTagName('Job')
from datetime import datetime

def fakeqstat(joblist):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match_string",
        default=None,
        type=str,
        # required=True,
        help="String to filter on",
    )
    args = parser.parse_args()

    data = []
    cancelled_data = []
    for r in joblist:
        try:
            # jobname=r.getElementsByTagName('JB_name')[0].childNodes[0].data
            # jobown=r.getElementsByTagName('JB_owner')[0].childNodes[0].data
            # jobstate=r.getElementsByTagName('state')[0].childNodes[0].data
            jobnum=r.getElementsByTagName('Job_Id')[0].childNodes[0].data.split(".")[0]
            jobname=r.getElementsByTagName('Job_Name')[0].childNodes[0].data
            jobstate=r.getElementsByTagName('job_state')[0].childNodes[0].data
            queue=r.getElementsByTagName('queue')[0].childNodes[0].data
            ctime=r.getElementsByTagName('ctime')[0].childNodes[0].data
            ctime = datetime.fromtimestamp(int(ctime)).strftime('%b-%d %H:%M')
            jobtime='not set'
            ftime='not set'

            with os.popen("/opt/moab/bin/showstart {jn}".format(jn = jobnum)) as g:
                lines = g.readlines()
            match = re.match('.* start in [^0-9-]+(-?[0-9]+.*)', lines[2])
            if match is not None:
                ETS = match.groups()[0]
            else:
                ETS = "None"

            if(jobstate=='R'):
                jobtime=r.getElementsByTagName('start_time')[0].childNodes[0].data
                jobtime = datetime.fromtimestamp(int(jobtime)).strftime('%b-%d %H:%M')
                ftime = r.getElementsByTagName('Walltime')[0].getElementsByTagName('Remaining')[0].childNodes[0].data
                ftime = datetime.fromtimestamp(time.time() + int(ftime)).strftime('%b-%d %H:%M')
            # elif(jobstate=='dt'):
            #     jobtime=r.getElementsByTagName('JAT_start_time')[0].childNodes[0].data
            # else:
            #     jobtime=r.getElementsByTagName('JB_submission_time')[0].childNodes[0].data
            elements_str = ['Job_Id','Job_Name','s','to_end','started','queue since','q','Estimated Start Time']
            elements = [jobnum,jobname,jobstate,ftime,jobtime,ctime,queue,ETS]
            if jobstate=='C':
                cancelled_data.append(elements)
            else:
                data.append(elements)
            # print('\t'.join([jobnum,jobname + '\t\t\t',jobstate]))#, '\t', jobown.ljust(16), '\t', jobname.ljust(16),'\t', jobstate,'\t',jobtime)
            # print(jobnum, '\t', jobown.ljust(16), '\t', jobname.ljust(16),'\t', jobstate,'\t',jobtime)
        except Exception as e:
            print(e)
    cancelled_data.reverse()
    data += cancelled_data
    if args.match_string is not None:
        print "Filtering on: ", args.match_string
        data = [el for el in data if args.match_string in el[1]]
    widths = [max(map(len, col)) for col in zip(*data)]
    for i, row in enumerate(data):
        if i == 0:
            print("  ".join((val.ljust(width) for val, width in zip(elements_str, widths))))
            print("  ".join(["-"*width for width in widths]))
        if i == len(data) - len(cancelled_data):
            print("  ".join(["-"*width for width in widths]))
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))
fakeqstat(jobs)
