# PretrainingRecommender

Source code for [*Knowledge Transfer via Pre-training for Recommendation: A Review and Prospect*](https://www.frontiersin.org/articles/10.3389/fdata.2021.602071/full)

We construct our verification experiment on [BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) and [caser_pytorch](https://github.com/graytowne/caser_pytorch).

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

