# PretrainingRecommender

Source code for [*Knowledge Transfer via Pre-training for Recommendation: A Review and Prospect*](https://www.frontiersin.org/articles/10.3389/fdata.2021.602071/full)

We construct our verification experiment on [BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) and [caser_pytorch](https://github.com/graytowne/caser_pytorch).

## Requirements
- torch==1.6.0
- scipy>=1.3.2
- Python3
- wget==3.2
- tqdm==4.36.1
- numpy==1.16.2
- tb-nightly==2.1.0a20191121
- pandas==0.25.0
- future==0.18.2


## BERT4Rec

Switch to metaBERT4Rec/ directory. 

Preprocessed data is in Data/ml-1m/, where a_ra.dat stands for ML-1m-src and c_ra.dat stands for ML-1m-tgt.

Set your parameters in `options.py`
(e.g. `--kg 1 --ifpre 1 --export MODEL_PATH` for pre-training on ML-1m-src with meta knowledge, and then `--kg 1 --ifpre 0 --ifcold 1 --full 1 --load MODEL_PATH` for fine-tuning with full/deep transfer on the cold-start ML-1m-tgt):

`python main.py --template train_bert`

## Caser

Switch to caser_pytorch/ directory.

Preprocessed data is in datasets/newdat/, where a_train(&eval&test).txt stands for ML-1m-src and c_train(&eval&test).txt stands for ML-1m-tgt.

Several arguments are the same as BERT4Rec, including `--kg`, `--export `, `--load` and `--full`. e.g. For mask learning:

```
python mlm_train.py --kg 1 --train_root 'datasets/newdat/a_train.txt --eval_root 'datasets/newdat/a_eval.txt --test_root 'datasets/newdat/a_test.txt' --nega_eval 'datasets/newdat/nega_sample_a_eval.pkl' --nega_test 'datasets/newdat/nega_sample_a_test.pkl' --export 'ckpt_mlm.pth' 
```

And then shallow transfer the model to ML-1m-tgt:

```
python train_caser.py --kg 1 --train_root 'datasets/newdat/c_train.txt --eval_root 'datasets/newdat/c_eval.txt --test_root 'datasets/newdat/c_test.txt' --nega_eval 'datasets/newdat/nega_sample_b_eval.pkl' --nega_test 'datasets/newdat/nega_sample_b_test.pkl' --load 'ckpt_mlm.pth' --full 0 --export 'ckpt_test.pth' 
```

## Experiment Result
![image](https://user-images.githubusercontent.com/40881407/119127644-28923380-ba67-11eb-8104-2c61be98876a.png)

## Cite
If you use the code, please cite this paper:
```
@article{zeng2021knowledge,
  title={Knowledge Transfer via Pre-training for Recommendation: A Review and Prospect},
  author={Zeng, Zheni and Xiao, Chaojun and Yao, Yuan and Xie, Ruobing and Liu, Zhiyuan and Lin, Fen and Lin, Leyu and Sun, Maosong},
  journal={Frontiers in big Data},
  volume={4},
  year={2021},
  publisher={Frontiers Media SA}
}
```


