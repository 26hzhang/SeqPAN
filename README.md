# Parallel Attention Network with Sequence Matching for Video Grounding

**TensorFlow** implementation for the paper "Parallel Attention Network with Sequence Matching for Video 
Grounding" (**ACL 2021 Findings**): [ArXiv version](https://arxiv.org/abs/2105.08481).

![overview](/figures/framework.png)

## Prerequisites
- python3 with tensorflow (>=`1.13.1`, <=`1.15.0`), tqdm, nltk, numpy, cuda10 and cudnn

## Preparation
The visual features of `Charades-STA`, `ActivityNet Captions` and `TACoS` are available at [Box Drive](
https://app.box.com/s/d7q5atlidb31cuj1u8znd7prgrck1r1s), download and place them under the `./data/features/` directory. 
Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it to 
`./data/features/` directory. Directory hierarchies are shown below:
```
SeqPAN
    |____ ckpt/
    |____ data/
        |____ datasets/
        |____ features/
            |____ activitynet/
            |____ charades/
            |____ tacos/
            |____ glove.840B.300d.txt
    ...
```

## Quick Start
**Train**
```shell script
# processed dataset will be automatically generated or loaded if exist
# set `--mode test` for evaluation
# train Charades-STA dataset
python main.py --task charades --max_pos_len 64 --char_dim 50 --mode train
# train ActivityNet Captions dataset
python main.py --task activitynet --max_pos_len 100 --char_dim 100 --mode train
# train TACoS dataset
python main.py --task tacos --max_pos_len 256 --char_dim 50 --mode train
```
**Test**
```shell script
# processed dataset will be automatically generated or loaded if exist
# set `--suffix xxx` to restore pre-trained parameters for evaluation
# where `xxx` denotes the name after the last `_` of the ckpt directory
# train Charades-STA dataset
python main.py --task charades --max_pos_len 64 --char_dim 50 --suffix xxx --mode test
# train ActivityNet Captions dataset
python main.py --task activitynet --max_pos_len 100 --char_dim 100 --suffix xxx --mode test
# train TACoS dataset
python main.py --task tacos --max_pos_len 256 --char_dim 50 --suffix xxx --mode test
```
You can also download the checkpoints for each task from [here](https://app.box.com/s/c302s8t2180ov1lkvx8n64ldcog8iyp6) 
and save them to the `./ckpt/` directory. The corresponding processed dataset is available at [here](
https://app.box.com/s/ya5k038r490w4nm10jzosd58o3tfb4vx), download and save them to the `./datasets/` directory. 
More hyper-parameter settings are in the `main.py`.

## Citation
If you feel this project helpful to your research, please cite our work.
```
@article{zhang2021parallel,
  title={Parallel Attention Network with Sequence Matching for Video Grounding},
  author={Zhang, Hao and Sun, Aixin and Jing, Wei and Zhen, Liangli and Zhou, Joey Tianyi and Goh, Rick Siow Mong},
  journal={arXiv preprint arXiv:2105.08481},
  year={2021}
}
```
