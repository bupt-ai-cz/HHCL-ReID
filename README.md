HHCL-ReID  (Coming Soon!)

This repository is the official implementation of our paper "Hard-sample Guided Hybrid Contrast Learning for Unsupervised Person Re-Identification!".  

![framework_HCCL](img/framework_HCCL.jpg)

Requirements

    git clone https://github.com/bupt-ai-cz/HHCL-ReID.git
    cd HHCL-ReID
    python setup.py develop

Prepare Datasets

Download the datasets Market-1501,MSMT17,DukeMTMC-reID and unzip them under the directory like:

    HHCL-ReID/examples/data
    ├── market1501
    │   └── Market-1501-v15.09.15
    ├── msmt17
    │   └── MSMT17_V1
    └── dukemtmcreid
        └── DukeMTMC-reID

Prepare ImageNet Pre-trained Models for IBN-Net

When training with the backbone of IBN-ResNet, you need to download the ImageNet-pretrained model from this link and save it under the path of examples/pretrained/.

Training

We utilize 4 GTX-2080TI GPUs for training. Examples:

Market-1501:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.45 --num-instances 16

MSMT17:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16

DukeMTMC-reID:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet50 -d dukemtmcreid --iter 200 --momentum 0.1 --eps 0.6 --num-instances 16

Evaluation

To evaluate my model on ImageNet, run:

    CUDA_VISIBLE_DEVICES=0 python examples/test.py -d $DATASET --resume $PATH

Results

Our model achieves the following performance on :

  Dataset           	Market1501	    	    	    	DukeMTMC-reID	    	    	    
  Setting           	mAP       	R1  	R5  	R10 	mAP          	R1  	R5  	R10 
  Fully Unsupervised	84.2      	93.4	97.7	98.5	73.3         	85.1	92.4	94.6
  Supervised        	87.2      	94.6	98.5	99.1	80.0         	89.8	95.2	96.7



Contributing

Pick a licence and describe how to contribute to your code repository. 

Acknowledgements

Thanks to Yixiao Ge for opening source of his excellent works SpCL.
