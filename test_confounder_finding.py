import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer

from constants import DEVLBERT_ROOT_DIR, OLD_10_100_LMDB_PATH, MTURK_DIR
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from util import CLASSES

os.chdir(DEVLBERT_ROOT_DIR)

from devlbert.datasets import ConceptCapLoaderTrain
from devlbert.devlbert import BertForMultiModalPreTraining, BertConfig

PRETRAINED_PATH = Path(DEVLBERT_ROOT_DIR, "save/devlbert/pytorch_model_11.bin")
CFG_PATH = Path(DEVLBERT_ROOT_DIR, "config/bert_base_6layer_6conect.json")
GROUND_TRUTH_PATH = Path(MTURK_DIR, 'output_mturk', 'pair_annotations_0.8_no_cfdnce_weight.tsv')
gt_for_pair = pd.read_csv(GROUND_TRUTH_PATH, sep='\t')


def main():
    model = BertForMultiModalPreTraining.from_pretrained(PRETRAINED_PATH, config=BertConfig.from_json_file(CFG_PATH))
    model.cuda()
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    train_dataset = ConceptCapLoaderTrain(
        tokenizer,
        lmdb_path=OLD_10_100_LMDB_PATH,
        seq_len=36,
        batch_size=2,
        predict_feature=False,
        distributed=False,
    )

    for batch in iter(train_dataset):
        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, \
        image_ids, causal_label_t, causal_label_v = batch

        # #For debugging: checking
        # model(
        #             input_ids,
        #             image_feat,
        #             image_loc,
        #             segment_ids,
        #             input_mask,
        #             image_mask,
        #             lm_label_ids,
        #             image_label,
        #             image_target,
        #             is_next,
        #             causal_label_t,
        #             causal_label_v)

        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = model.bert(
            input_ids,
            image_feat,
            image_loc,
            segment_ids,
            input_mask,
            image_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=None
        )

        y = sequence_output_v[:, 1:].cuda()
        for b_idx, boxes in enumerate(y):
            attention = torch.mm(model.causal_v.Wy(boxes), model.causal_v.Wz(model.causal_v.dic_z).t()) / (
                    model.causal_v.embedding_size ** 0.5)
            attention = F.softmax(attention, 1)  # torch.Size([box, 1601])
            # Most-matched element in the confounder dictionary is claimed to be most likely class to form
            # a confounder for ?r and r?
            _, max_attented_ids = attention.max(-1)
            _, max_box_ids = image_target[b_idx].max(-1)
            max_attented_classes = [CLASSES[i] for i in max_attented_ids.tolist()]
            max_box_classes = [CLASSES[i] for i in max_box_ids.tolist()]
            confounders_for_objects_for_id = (int(image_ids[b_idx]), [(b, a) for b, a, m in
                                                                      zip(max_box_classes, max_attented_classes,
                                                                          image_mask[b_idx]) if m != 0])
            sorted_cause_scores, sorted_causes_ids = attention.sort(-1, descending=True)
            for box_idx, effect_object in enumerate(max_box_classes):
                known_gts = gt_for_pair[
                    (effect_object == gt_for_pair['word_X']) | (effect_object == gt_for_pair['word_Y'])]
                if len(known_gts) == 0:
                    continue
                pred_causes = [(CLASSES[i], float(score)) for i, score in
                               zip(sorted_causes_ids[box_idx].tolist(), sorted_cause_scores[box_idx])]
                known_gts['cause_candidate'] = known_gts['word_X'].where(known_gts['word_Y'] == effect_object,
                                                                         known_gts['word_Y'])
                known_gts['cause_candidate_label'] = np.where(
                    (known_gts['word_X'] == known_gts['cause_candidate']) & (known_gts['max_resp'] == 'x-to-y'),
                    'cause', np.where(known_gts['max_resp'] == 'z-to-xy', 'mere_correlate', 'effect'))
                print(list(zip(known_gts.cause_candidate, known_gts.cause_candidate_label,
                               [dict(pred_causes)[w] for w in known_gts.cause_candidate])))

                # Treating confounder-attention as a ranking, where we want the ground-truth causes in our to be
                # ranked higher than the ground-truth-not-causes
                result = sorted(list(zip(known_gts.cause_candidate, known_gts.cause_candidate_label,
                                         [dict(pred_causes)[w] for w in known_gts.cause_candidate])),
                                key=lambda r: r[-1], reverse=True)

                def rel_at_k(k):
                    return int(result[k][1] == 'cause')

                def precision_at_k(k):
                    return len([row for row in result[:k + 1] if row[1] == 'cause']) / (k + 1)

                def average_precision():
                    if len([row for row in result if row[1] == 'cause']) != 0:
                        return sum([precision_at_k(k) * rel_at_k(k) for k in range(len(result))]) / len(
                            [row for row in result if row[1] == 'cause'])
                    else:
                        return None
                nb_causes = len([r for r in result if r[1] == 'cause'])
                expected_average_precision = None #TODO implement exact value from https://ufal.mff.cuni.cz/pbml/103/art-bestgen.pdf instead of just proportion of causes
                if nb_causes == 0:
                    print(f"No causes in database for {effect_object}")
                else:
                    print(average_precision() - expected_average_precision)
            print(confounders_for_objects_for_id)


if __name__ == '__main__':
    main()
