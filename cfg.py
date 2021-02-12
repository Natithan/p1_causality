import sys
from absl import flags
FGS = flags.FLAGS
flags.DEFINE_list("pixel_mean", [102.9801, 115.9465, 122.7717],"")
flags.DEFINE_string("image_dir","/cw/liir/NoCsBack/testliir/datasets/ConceptualCaptions/training", "")
flags.DEFINE_integer("max_nb_images",1000, "")
flags.DEFINE_integer("batch_size",8, "")
flags.DEFINE_integer("num-cpus",32, "")
flags.DEFINE_string("gpus","1,2,3", "")
flags.DEFINE_list("min-max-boxes",[0,1,2,3], "")
flags.DEFINE_string("extract-mode","roi_feats", "")
flags.DEFINE_string("config-file","buatest/configs/bua-caffe/extract-bua-caffe-r101.yaml", "")
flags.DEFINE_string("image-dir","/cw/liir/NoCsBack/testliir/datasets/ConceptualCaptions/training", "")
flags.DEFINE_string("mode", "caffe", "Flag to allow python console command line argument")
flags.DEFINE_bool("verbose",False, "")
FGS(sys.argv)