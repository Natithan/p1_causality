from time import time

from pathlib import Path


ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan/p1_causality"

import sys
from absl import flags
FGS = flags.FLAGS
flags.DEFINE_integer("checkpoint_period", 60 * 60, "Number seconds between each time a checkpoint is stored")

FGS(sys.argv)
