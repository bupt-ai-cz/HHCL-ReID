### resnet50 ###
# market1501
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet50 -d market1501 --iters 200 --eps 0.45 --num-instances 16 --pooling-type avg --memorybank CMhybird --epochs 60 --logs-dir examples/logs/market1501/resnet50_avg_cmhybird

# dukemtmcreid
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --eps 0.6 --num-instances 16 --pooling-type avg --memorybank CMhybird --epochs 60 --logs-dir examples/logs/dukemtmcreid/resnet50_avg_cmhybird


### resnet_ibn50a + gem pooling ###
# market1501
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet_ibn50a -d market1501 --iters 200 --eps 0.45 --num-instances 16 --pooling-type gem --memorybank CMhybird --epochs 60 --logs-dir examples/logs/market1501/resnet50_ibn_gem_cmhybird

# dukemtmcreid
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet_ibn50a -d dukemtmcreid --iters 200 --eps 0.6 --num-instances 16 --pooling-type gem --memorybank CMhybird --epochs 60 --logs-dir examples/logs/dukemtmcreid/resnet50_ibn_gem_cmhybird

# test
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d market1501 --data-dir examples/data/market1501 --pooling-type avg --resume examples/logs/market1501/resnet50_avg_cmhybird/model_best.pth.tar