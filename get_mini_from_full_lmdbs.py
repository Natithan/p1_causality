import lmdb
import tqdm
import os

FULL_SIZE = 3000000 # Just about
size_for_name = {
    'mini': FULL_SIZE // 1000,
    'centi': FULL_SIZE // 100,
}
CURRENT_SIZE = 'centi'

def put_or_grow(ev, tx, key, value):
    try:
        tx.put(key, value)
        return tx
    except lmdb.MapFullError:
        pass
    tx.abort()
    curr_size = ev.info()['map_size']
    new_size = curr_size * 2
    print("Doubling LMDB map_size to {:.2f}GB".format(new_size / 10 ** 9))
    ev.set_mapsize(new_size)
    tx = ev.begin(write=True)
    tx = put_or_grow(ev, tx, key, value)
    return tx


p_full = f"/cw/working-gimli/nathan/features_CoCa_lmdb/full_coca_36_0_of_4.lmdb"
env_full = lmdb.open(p_full, readonly=True, subdir=os.path.isdir(p_full))
txn_full = env_full.begin()
vs = []
for i, v in enumerate(txn_full.cursor()):
    if i > size_for_name[CURRENT_SIZE]:
        break
    vs.append(v)

for j in range(4):
    p_mini = f"/cw/working-gimli/nathan/features_CoCa_lmdb/{CURRENT_SIZE}_coca_36_{j}_of_4.lmdb"
    assert p_full != p_mini

    env_mini = lmdb.open(p_mini, subdir=False, readonly=False, map_size=1099511627776 * 2)
    txn_mini = env_mini.begin(write=True)

    for v in tqdm.tqdm(vs[j::4]):
        txn_mini = put_or_grow(env_mini, txn_mini, *v)
    txn_mini.commit()
