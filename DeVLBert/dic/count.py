import lmdb
import numpy as np
from multiprocessing import Process, Queue
from pytorch_pretrained_bert.tokenization import BertTokenizer
import tensorpack.dataflow as td
import json
import collections
import nltk
import os
from constants import LMDB_PATH
from tqdm import tqdm

def run_count(q, num):
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )
    with open('noun_set.json', 'r') as f:
        se = set(json.load(f))
    ds = td.LMDBSerializer.load(LMDB_PATH, shuffle=False)
    num_dataset = len(ds)
    print(num_dataset)
    ds.reset_state()
    dic = {}
    for i, batch in tqdm(enumerate(ds.get_data(), 1),total=num_dataset):
        image_feature_wp, image_target_wp, image_location_wp, num_boxes, image_h, image_w, image_id, caption = batch
        tokens = tokenizer.tokenize(caption)
        while len(tokens) > 34:
            tokens.pop()
        l = len(tokens)
        assert l <= 34
        for j in range(l):
            if tokens[j] in se and tokens[j][0] != "#" and (j != l-1 and tokens[j+1][0] != "#" or j == l-1):
                dic[tokens[j]] = dic.get(tokens[j], 0) + 1

    q.put(dic)
    print("finish --- process {}".format(num))


def distribute(nb_processes=None):
    if nb_processes is None:
        nb_processes = os.cpu_count()
    pool = []
    q = Queue()
    for i in range(nb_processes):
        process = Process(target=run_count, args=(q, i))
        pool.append(process)
    for process in pool:
        process.start()
    arr = []
    for i in range(nb_processes):
        dic = q.get()
        arr.append(dic)
    for process in pool:
        process.join()
    print("join Done")
    return arr


if __name__=='__main__':
    arr = distribute(1)

    res = {}
    for dic in arr:
        for k, v in dic.items():
            res[k] = res.get(k, 0) + v

    res = sorted(res.items(), key=lambda x: x[1], reverse=True)
    d = collections.OrderedDict()
    for tup in res:
        d[tup[0]] = tup[1]

    with open('noun_frequency.json', 'w') as f:
        json.dump(d, f, indent=4)