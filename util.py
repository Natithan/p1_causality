import os
import pandas as pd
import ray

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import numpy as np

from tools.DownloadConcptualCaption.download_data import _file_name

with open("/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert/dic/objects_vocab.txt", "r") as vocab:
    CLASSES = ['background'] + [line.strip() for line in vocab]

IMAGE_DIR = "/cw/liir/NoCsBack/testliir/datasets/ConceptualCaptions/training"


def show_from_tuple(rpn_tuple):
    _, cls_probs, bboxes, _, _, _, img_id, caption = rpn_tuple

    image = Image.open(os.path.join(IMAGE_DIR, img_id))
    plt.imshow(np.asarray(image))
    # Get the current reference
    ax = plt.gca()  # Create a Rectangle patch
    colors = ['black', 'darkblue', 'darkmagenta']
    for b, probs in zip(bboxes, cls_probs):
        color = random.choice(colors)
        rect = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1, edgecolor=color, facecolor='none')
        cls = CLASSES[np.argmax(probs)]
        ax.text(b[0],b[1] - 12 / 2, cls, color=color)
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()


@ray.remote
def index_df_column(dataframe, df_column):
    col_for_ids = {}
    for i, img in enumerate(dataframe.iterrows()):
        col = img[1][df_column]  # .decode("utf8")
        img_name = _file_name(img[1])
        image_id = img_name.split('/')[1]
        # image_id = str(i)
        col_for_ids[image_id] = col
    return col_for_ids


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption", "url"], usecols=range(0, 2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df