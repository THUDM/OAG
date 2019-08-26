# OAG

OAG: Toward Linking Large-scale Heterogeneous Entity Graphs.<br>
Fanjin Zhang, Xiao Liu, Jie Tang, Yuxiao Dong, Peiran Yao, Jie Zhang, Xiaotao Gu, Yan Wang, Bin Shao, Rui Li, and Kuansan Wang.<br>
In KDD 2019 (Applied Data Science Track)

## Prerequisites

- Linux or macOS
- Python 3
- TensorFlow GPU >= 1.14
- NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUKG/OAG
cd OAG
```

Please install dependencies by

```bash
pip install -r requirements.txt
```
### Dataset

The dataset can be downloaded from [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:u:/g/personal/zfj17_mails_tsinghua_edu_cn/ES2s-PhyDeREs1zk0qdnA08BhzBZRSzrzKCqGAjEvdGBVQ?e=6U3bOd), [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/1141adb4aac240d7a49d/?dl=1) or [BaiduPan](https://pan.baidu.com/s/1ZkIs89yy9TrDMssZ3ceeVw) (with password gzpp). Unzip the file and put the _data_ directory into project directory.

## How to run
```bash
cd $project_path
export PYTHONPATH="$project_path:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
cd core

# venue linking
python rnn/train.py

# paper linking
### LSH method
python hash/title2vec.py  # train doc2vec model
python hash/hash.py
### CNN method
python cnn/train.py

# author linking
python gat/preprocessing.py
python gat/train.py
```

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{zhang2019oag,
  title={OAG: Toward Linking Large-scale Heterogeneous Entity Graphs.},
  author={Zhang, Fanjin and Liu, Xiao and Tang, Jie and Dong, Yuxiao and Yao, Peiran and Zhang, Jie and Gu, Xiaotao and Wang, Yan and Shao, Bin and Li, Rui and Wang, Kuansan},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD’19)},
  year={2019}
}
```
