import time

import numpy as np
import os
ROOT = '/cw/liir/NoCsBack/testliir/nathan/p1_causality'
c = np.load(os.path.join(ROOT, "DeVLBert/dic", "id2class.npy"), allow_pickle=True).item()
c1155_mine = np.load(os.path.join(ROOT, "DeVLBert/dic", "id2class1155_mine.npy"), allow_pickle=True).item()
c1155_og = np.load(os.path.join(ROOT, "DeVLBert/dic", "id2class1155.npy"), allow_pickle=True).item()
from pytorch_pretrained_bert.tokenization import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True
)
id2word = {v: k for k, v in tokenizer.vocab.items()}
nouns = [id2word[i] for i in c.keys()]
nouns1155 = [id2word[i] for i in c1155_mine.keys()]
nouns1155_og = [id2word[i] for i in c1155_og.keys()]
objects = [o[:-1] for o in open(os.path.join(ROOT, "DeVLBert/dic","objects_vocab.txt"), "r")]
import bnlearn as bn

gt_dag_names_ex = ['titanic', 'sprinkler', 'alarm', 'andes', 'asia', 'pathfinder', 'sachs', 'water']
gt_dag_names_dag = ['sprinkler', 'alarm', 'andes', 'asia', 'pathfinder', 'sachs', 'miserables']
import signal
def handler(signum, frame):
     print("Forever is over!")
     raise Exception("end of time")
signal.signal(signal.SIGALRM, handler)


# examples_dict = {k: list(bn.import_example(k).columns) for k in gt_dag_names_ex}
examples_dict = {}
for k in gt_dag_names_ex:
    print(k)
    signal.alarm(2)
    try:
        examples_dict[k] = list(bn.import_example(k).columns)
        signal.alarm(0)
    except Exception as e:
        print(e)
        signal.alarm(0)
        continue

DAG_dict = {}
for k in gt_dag_names_dag:
    print(k)
    s = time.time()
    DAG_dict[k] = list(bn.import_DAG(k,CPD=False)['model'].nodes)
    print(time.time() - s)
