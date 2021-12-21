import os
filestring = ""
pbs_args = [
    ("A", 'lcalculus'),
    ("l", 'nodes=1:ppn=36:gpus=4:skylake'),
    ("l", 'partition=gpu'),
    ("l", 'pmem=5gb'),
    ("m", 'abe'),
    ("j", 'nathan.cornille@kuleuven.be'),
    ("M", 'oe'),
]
filename = "test.pbs"
with open(filename,'w') as f:
    f.write(filestring)
# os.system(f"qsub {filename}")
#TODO