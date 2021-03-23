from pathlib import Path
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer

from constants import DEVLBERT_ROOT_DIR, OLD_10_100_LMDB_PATH
import os

from util import CLASSES

os.chdir(DEVLBERT_ROOT_DIR)

from devlbert.datasets import ConceptCapLoaderTrain
from devlbert.devlbert import BertForMultiModalPreTraining, BertConfig

PRETRAINED_PATH = Path(DEVLBERT_ROOT_DIR, "save/devlbert/pytorch_model_11.bin")
CFG_PATH = Path(DEVLBERT_ROOT_DIR, "config/bert_base_6layer_6conect.json")


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

    input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, \
    image_ids, causal_label_t, causal_label_v = next(iter(train_dataset))

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
    for b_idx,boxes in enumerate(y):
        attention = torch.mm(model.causal_v.Wy(boxes), model.causal_v.Wz(model.causal_v.dic_z).t()) / (
                    model.causal_v.embedding_size ** 0.5)
        attention = F.softmax(attention, 1)  # torch.Size([box, 1601])
        # Most-matched element in the confounder dictionary is claimed to be most likely class to form
        # a confounder for ?r and r? #TODO figure out how to deal with the model D quirk
        _,max_attented_ids = attention.max(-1)
        _, max_box_ids = image_target[b_idx].max(-1)
        max_attented_classes = [CLASSES[i] for i in max_attented_ids.tolist()]
        max_box_classes = [CLASSES[i] for i in max_box_ids.tolist()]
        confounders_for_objects_for_id = (int(image_ids[b_idx]),[(b,a )for b,a,m in zip(max_box_classes,max_attented_classes,image_mask[b_idx]) if m != 0])


if __name__ == '__main__':
    main()
