## clone the repository including Detectron2(@be792b9)
#git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch
#cd "bottom-up-attention.pytorch"
#cd detectron2
#pip install -e .
#cd ..
#
## install apex
#git clone https://github.com/NVIDIA/apex.git
#cd apex
#python setup.py install
#cd ..
##install the rest modules
#python setup.py build develop
#pip install ray


# Get specific version of torch
pip install --upgrade torch
pip install --upgrade torchvision
# clone the repository including Detectron2(@be792b9)
git clone https://github.com/MILVLG/bottom-up-attention.pytorch
cd "bottom-up-attention.pytorch"
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..

# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..
# install the rest modules
python setup.py build develop
pip install ray
